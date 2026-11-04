//! `.pth` checkpoints written by `torch.save`: a zip archive (PyTorch >= 1.6) or the legacy
//! sequence of pickles followed by the raw storages.
//!
//! The pickle is read by a restricted unpickler that only builds containers, scalars and tensor
//! references: nothing from the file is executed, and any global or opcode outside what weight
//! files use is an error naming it.

use super::{DType, TensorF32, decode};
use scaleup_core::{Result, ScaleUpError};
use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};
use std::io::{Cursor, Read};
use std::path::Path;
use std::rc::Rc;

type PResult<T> = std::result::Result<T, String>;

/// Dicts that wrap the real state dict in training checkpoints, tried in this order (spandrel's
/// `canonicalize_state_dict`).
const WRAPPER_KEYS: [&str; 7] = [
    "model_state_dict",
    "state_dict",
    "params_ema",
    "params-ema",
    "params",
    "model",
    "net",
];

/// First pickle of a legacy file.
const LEGACY_MAGIC: i128 = 0x1950a86a20f9469cfc6c;
const LEGACY_PROTOCOL: i128 = 1001;

pub(super) fn load(path: &Path) -> Result<BTreeMap<String, TensorF32>> {
    let bytes = std::fs::read(path)?;
    let parsed = if bytes.starts_with(b"PK\x03\x04") {
        load_zip(&bytes)
    } else if bytes.first() == Some(&0x80) {
        load_legacy(&bytes)
    } else {
        Err("not a PyTorch checkpoint (neither a zip archive nor a pickle)".to_string())
    };
    parsed.map_err(|e| ScaleUpError::Model(format!("{}: {e}", path.display())))
}

fn load_zip(bytes: &[u8]) -> PResult<BTreeMap<String, TensorF32>> {
    let mut zip = zip::ZipArchive::new(Cursor::new(bytes)).map_err(|e| format!("zip: {e}"))?;
    // Entries live under one top-level directory whose name varies (`archive`, the file stem...).
    let pickle_name = zip
        .file_names()
        .filter_map(|n| n.ok())
        .find(|n| n.ends_with("/data.pkl") && n.matches('/').count() == 1)
        .ok_or("zip archive has no <dir>/data.pkl")?
        .into_owned();
    let root = &pickle_name[..pickle_name.len() - "data.pkl".len()];
    if let Ok(order) = read_entry(&mut zip, &format!("{root}byteorder"))
        && order != b"little"
    {
        return Err(format!(
            "byte order {:?} is not supported",
            String::from_utf8_lossy(&order)
        ));
    }
    let pickle = read_entry(&mut zip, &pickle_name)?;
    let mut unpickler = Unpickler::new(&pickle);
    let root_value = unpickler.load()?;
    let tensors = state_dict_tensors(root_value)?;

    let mut storages: HashMap<String, Vec<u8>> = HashMap::new();
    let mut out = BTreeMap::new();
    for (key, t) in tensors {
        let skey = &t.storage.key;
        if !storages.contains_key(skey) {
            let raw = read_entry(&mut zip, &format!("{root}data/{skey}"))?;
            storages.insert(skey.clone(), raw);
        }
        out.insert(key, materialize(&t, &storages[skey])?);
    }
    Ok(out)
}

fn read_entry(zip: &mut zip::ZipArchive<Cursor<&[u8]>>, name: &str) -> PResult<Vec<u8>> {
    let mut entry = zip
        .by_name(name)
        .map_err(|e| format!("zip entry {name}: {e}"))?;
    let mut buf = Vec::new();
    entry
        .read_to_end(&mut buf)
        .map_err(|e| format!("zip entry {name}: {e}"))?;
    Ok(buf)
}

/// Legacy layout: pickles of the magic number, the protocol version, the system info, the object
/// and the list of storage keys; then, for each key in that order, its element count (i64 LE)
/// and its raw data.
fn load_legacy(bytes: &[u8]) -> PResult<BTreeMap<String, TensorF32>> {
    let mut pos = 0;
    let mut next = |what: &str| -> PResult<(Value, HashMap<String, Rc<StorageRef>>)> {
        let mut u = Unpickler::new(&bytes[pos..]);
        let v = u.load().map_err(|e| format!("legacy {what}: {e}"))?;
        pos += u.pos;
        Ok((v, u.storages))
    };
    if !matches!(next("magic number")?.0, Value::Int(LEGACY_MAGIC)) {
        return Err("not a PyTorch checkpoint (bad legacy magic number)".to_string());
    }
    match next("protocol version")?.0 {
        Value::Int(LEGACY_PROTOCOL) => {}
        other => {
            return Err(format!(
                "legacy protocol version {other:?} is not supported"
            ));
        }
    }
    let Value::Dict(info) = next("system info")?.0 else {
        return Err("legacy system info is not a dict".to_string());
    };
    if !matches!(
        dict_get(&info.borrow(), "little_endian"),
        Some(Value::Bool(true))
    ) {
        return Err("legacy checkpoint is not little-endian".to_string());
    }
    let (root_value, storages) = next("object")?;
    let Value::List(keys) = next("storage keys")?.0 else {
        return Err("legacy storage key list is not a list".to_string());
    };

    let mut raw: HashMap<String, &[u8]> = HashMap::new();
    for key in keys.borrow().iter() {
        let Value::Str(key) = key else {
            return Err(format!("legacy storage key is a {}", key.kind()));
        };
        let storage = storages
            .get(key.as_ref())
            .ok_or_else(|| format!("legacy storage {key} is listed but never referenced"))?;
        let count = bytes
            .get(pos..pos + 8)
            .ok_or_else(|| format!("legacy storage {key}: truncated header"))?;
        let count = i64::from_le_bytes(count.try_into().unwrap());
        if usize::try_from(count).ok() != Some(storage.numel) {
            return Err(format!(
                "legacy storage {key} holds {count} elements, its tensors expect {}",
                storage.numel
            ));
        }
        pos += 8;
        let end = storage
            .byte_len()?
            .checked_add(pos)
            .filter(|&end| end <= bytes.len())
            .ok_or_else(|| format!("legacy storage {key}: truncated data"))?;
        raw.insert(key.to_string(), &bytes[pos..end]);
        pos = end;
    }

    let mut out = BTreeMap::new();
    for (key, t) in state_dict_tensors(root_value)? {
        let data = raw
            .get(&t.storage.key)
            .ok_or_else(|| format!("storage {} of {key} is missing", t.storage.key))?;
        out.insert(key, materialize(&t, data)?);
    }
    Ok(out)
}

/// Finds the state dict in the unpickled object (unwrapping a training-checkpoint wrapper) and
/// returns its tensors; entries that are not tensors (step counters, names, options) are skipped.
fn state_dict_tensors(root: Value) -> PResult<Vec<(String, Rc<TensorRef>)>> {
    let Value::Dict(mut dict) = root else {
        return Err(format!(
            "the checkpoint holds a {}, not a dict",
            root.kind()
        ));
    };
    let wrapped = WRAPPER_KEYS
        .iter()
        .find_map(|k| match dict_get(&dict.borrow(), k) {
            Some(Value::Dict(inner)) => Some(inner.clone()),
            _ => None,
        });
    if let Some(inner) = wrapped {
        dict = inner;
    }
    let single = match dict.borrow().as_slice() {
        [(_, Value::Dict(inner))] => Some(inner.clone()),
        _ => None,
    };
    if let Some(inner) = single {
        dict = inner;
    }

    let mut out = Vec::new();
    for (k, v) in dict.borrow().iter() {
        let Value::Tensor(t) = v else { continue };
        let Value::Str(name) = k else {
            return Err(format!("a tensor is stored under a {} key", k.kind()));
        };
        out.push((name.to_string(), t.clone()));
    }
    Ok(out)
}

/// The last value set for `key`, as a Python dict would keep.
fn dict_get<'a>(dict: &'a [(Value, Value)], key: &str) -> Option<&'a Value> {
    dict.iter()
        .rev()
        .find(|(k, _)| matches!(k, Value::Str(s) if s.as_ref() == key))
        .map(|(_, v)| v)
}

/// Gathers a (possibly strided or offset) view of a storage into a dense row-major tensor.
fn materialize(t: &TensorRef, raw: &[u8]) -> PResult<TensorF32> {
    let s = &t.storage;
    let expected = s.byte_len()?;
    if raw.len() != expected {
        return Err(format!(
            "storage {} holds {} bytes, expected {expected}",
            s.key,
            raw.len()
        ));
    }
    let size = s.dtype.size();
    let numel: usize = t.shape.iter().product();
    let mut data = Vec::with_capacity(numel);
    let mut index = vec![0usize; t.shape.len()];
    for _ in 0..numel {
        let element = t.offset
            + index
                .iter()
                .zip(&t.stride)
                .map(|(i, st)| i * st)
                .sum::<usize>();
        let at = element * size;
        data.push(decode(s.dtype, &raw[at..at + size]));
        for d in (0..index.len()).rev() {
            index[d] += 1;
            if index[d] < t.shape[d] {
                break;
            }
            index[d] = 0;
        }
    }
    Ok(TensorF32 {
        shape: t.shape.clone(),
        data,
    })
}

#[derive(Debug)]
struct StorageRef {
    dtype: DType,
    key: String,
    numel: usize,
}

impl StorageRef {
    fn byte_len(&self) -> PResult<usize> {
        self.numel
            .checked_mul(self.dtype.size())
            .ok_or_else(|| format!("storage {} is too large", self.key))
    }
}

#[derive(Debug)]
struct TensorRef {
    storage: Rc<StorageRef>,
    offset: usize,
    shape: Vec<usize>,
    stride: Vec<usize>,
}

/// The callables a weight file may name. Calling one only builds a value.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Global {
    OrderedDict,
    RebuildTensorV2,
    RebuildParameter,
    StorageType(DType),
}

impl Global {
    fn resolve(module: &str, name: &str) -> PResult<Global> {
        Ok(match (module, name) {
            ("collections", "OrderedDict") => Global::OrderedDict,
            ("torch._utils", "_rebuild_tensor_v2") => Global::RebuildTensorV2,
            ("torch._utils", "_rebuild_parameter") => Global::RebuildParameter,
            ("torch", "FloatStorage") => Global::StorageType(DType::F32),
            ("torch", "HalfStorage") => Global::StorageType(DType::F16),
            ("torch", "BFloat16Storage") => Global::StorageType(DType::BF16),
            ("torch", "DoubleStorage") => Global::StorageType(DType::F64),
            ("torch", "LongStorage") => Global::StorageType(DType::I64),
            ("torch", "IntStorage") => Global::StorageType(DType::I32),
            _ => return Err(format!("unsupported pickle global {module}.{name}")),
        })
    }
}

type ListRef = Rc<RefCell<Vec<Value>>>;
type DictRef = Rc<RefCell<Vec<(Value, Value)>>>;

/// Dicts and lists are shared so that a memoized container sees later `SETITEM`s, as in Python.
#[derive(Debug, Clone)]
enum Value {
    None,
    Bool(bool),
    Int(i128),
    /// Only options and metadata are floats; their value is never needed.
    Float,
    Str(Rc<str>),
    Tuple(Rc<[Value]>),
    List(ListRef),
    Dict(DictRef),
    Global(Global),
    Storage(Rc<StorageRef>),
    Tensor(Rc<TensorRef>),
}

impl Value {
    fn kind(&self) -> &'static str {
        match self {
            Value::None => "None",
            Value::Bool(_) => "bool",
            Value::Int(_) => "int",
            Value::Float => "float",
            Value::Str(_) => "str",
            Value::Tuple(_) => "tuple",
            Value::List(_) => "list",
            Value::Dict(_) => "dict",
            Value::Global(_) => "global",
            Value::Storage(_) => "storage",
            Value::Tensor(_) => "tensor",
        }
    }

    fn new_dict() -> Value {
        Value::Dict(Rc::new(RefCell::new(Vec::new())))
    }

    fn as_usize(&self, what: &str) -> PResult<usize> {
        match self {
            Value::Int(i) => usize::try_from(*i).map_err(|_| format!("{what} {i} is out of range")),
            other => Err(format!("{what} is a {}, not an int", other.kind())),
        }
    }

    fn as_usizes(&self, what: &str) -> PResult<Vec<usize>> {
        match self {
            Value::Tuple(items) => items.iter().map(|v| v.as_usize(what)).collect(),
            other => Err(format!("{what} is a {}, not a tuple", other.kind())),
        }
    }
}

struct Unpickler<'a> {
    data: &'a [u8],
    pos: usize,
    stack: Vec<Value>,
    marks: Vec<usize>,
    memo: HashMap<u32, Value>,
    /// Every storage referenced so far, by key.
    storages: HashMap<String, Rc<StorageRef>>,
}

impl<'a> Unpickler<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            pos: 0,
            stack: Vec::new(),
            marks: Vec::new(),
            memo: HashMap::new(),
            storages: HashMap::new(),
        }
    }

    fn take(&mut self, n: usize) -> PResult<&'a [u8]> {
        let bytes = self
            .data
            .get(self.pos..self.pos.saturating_add(n))
            .ok_or_else(|| format!("pickle truncated at byte {}", self.pos))?;
        self.pos += n;
        Ok(bytes)
    }

    fn u8(&mut self) -> PResult<u8> {
        Ok(self.take(1)?[0])
    }

    fn u32(&mut self) -> PResult<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn line(&mut self) -> PResult<&'a str> {
        let rest = &self.data[self.pos..];
        let end = rest
            .iter()
            .position(|&b| b == b'\n')
            .ok_or("unterminated GLOBAL name")?;
        self.pos += end + 1;
        std::str::from_utf8(&rest[..end]).map_err(|_| "GLOBAL name is not UTF-8".to_string())
    }

    fn string(&mut self, len: usize) -> PResult<Value> {
        let bytes = self.take(len)?;
        let s = std::str::from_utf8(bytes).map_err(|_| "string is not UTF-8".to_string())?;
        Ok(Value::Str(s.into()))
    }

    fn pop(&mut self) -> PResult<Value> {
        if self.marks.last().is_some_and(|&m| m >= self.stack.len()) {
            return Err(format!("pickle pops past a MARK at byte {}", self.pos));
        }
        self.stack
            .pop()
            .ok_or_else(|| format!("pickle stack underflow at byte {}", self.pos))
    }

    fn pop_mark(&mut self) -> PResult<Vec<Value>> {
        let mark = self
            .marks
            .pop()
            .ok_or_else(|| format!("pickle has no MARK at byte {}", self.pos))?;
        Ok(self.stack.split_off(mark))
    }

    /// Like Python's unpickler, values below the last MARK are out of reach.
    fn top(&self) -> PResult<&Value> {
        let floor = self.marks.last().copied().unwrap_or(0);
        self.stack[floor..]
            .last()
            .ok_or_else(|| format!("pickle stack underflow at byte {}", self.pos))
    }

    fn top_dict(&self) -> PResult<DictRef> {
        match self.top()? {
            Value::Dict(d) => Ok(d.clone()),
            other => Err(format!(
                "SETITEM on a {} at byte {}",
                other.kind(),
                self.pos
            )),
        }
    }

    fn top_list(&self) -> PResult<ListRef> {
        match self.top()? {
            Value::List(l) => Ok(l.clone()),
            other => Err(format!("APPEND on a {} at byte {}", other.kind(), self.pos)),
        }
    }

    fn memo_put(&mut self, index: u32) -> PResult<()> {
        let v = self.top()?.clone();
        self.memo.insert(index, v);
        Ok(())
    }

    fn memo_get(&mut self, index: u32) -> PResult<()> {
        let v = self
            .memo
            .get(&index)
            .ok_or_else(|| format!("pickle memo entry {index} is missing"))?
            .clone();
        self.stack.push(v);
        Ok(())
    }

    /// Runs the pickle up to `STOP` and returns the object it built.
    fn load(&mut self) -> PResult<Value> {
        loop {
            let at = self.pos;
            let op = self.u8()?;
            match op {
                0x80 => {
                    let proto = self.u8()?;
                    if proto > 5 {
                        return Err(format!("pickle protocol {proto} is not supported"));
                    }
                }
                0x95 => {
                    self.take(8)?; // FRAME: a length hint
                }
                b'.' => {
                    let v = self.pop()?;
                    if !self.stack.is_empty() || !self.marks.is_empty() {
                        return Err("pickle leaves values on the stack at STOP".to_string());
                    }
                    return Ok(v);
                }
                b'(' => self.marks.push(self.stack.len()),
                b'}' => self.stack.push(Value::new_dict()),
                b']' => self
                    .stack
                    .push(Value::List(Rc::new(RefCell::new(Vec::new())))),
                b')' => self.stack.push(Value::Tuple(Rc::from([]))),
                b't' => {
                    let items = self.pop_mark()?;
                    self.stack.push(Value::Tuple(items.into()));
                }
                0x85..=0x87 => {
                    let n = usize::from(op - 0x84);
                    let mut items = Vec::with_capacity(n);
                    for _ in 0..n {
                        items.push(self.pop()?);
                    }
                    items.reverse();
                    self.stack.push(Value::Tuple(items.into()));
                }
                b'q' => {
                    let i = self.u8()?;
                    self.memo_put(u32::from(i))?;
                }
                b'r' => {
                    let i = self.u32()?;
                    self.memo_put(i)?;
                }
                0x94 => {
                    let i = u32::try_from(self.memo.len()).map_err(|_| "memo overflow")?;
                    self.memo_put(i)?;
                }
                b'h' => {
                    let i = self.u8()?;
                    self.memo_get(u32::from(i))?;
                }
                b'j' => {
                    let i = self.u32()?;
                    self.memo_get(i)?;
                }
                b'J' => {
                    let v = i32::from_le_bytes(self.take(4)?.try_into().unwrap());
                    self.stack.push(Value::Int(i128::from(v)));
                }
                b'K' => {
                    let v = self.u8()?;
                    self.stack.push(Value::Int(i128::from(v)));
                }
                b'M' => {
                    let v = u16::from_le_bytes(self.take(2)?.try_into().unwrap());
                    self.stack.push(Value::Int(i128::from(v)));
                }
                0x8a => {
                    let n = usize::from(self.u8()?);
                    if n > 16 {
                        return Err(format!("LONG1 of {n} bytes at byte {at} is too large"));
                    }
                    let bytes = self.take(n)?;
                    let negative = bytes.last().is_some_and(|b| b & 0x80 != 0);
                    let mut buf = [if negative { 0xff } else { 0 }; 16];
                    buf[..n].copy_from_slice(bytes);
                    self.stack.push(Value::Int(i128::from_le_bytes(buf)));
                }
                b'G' => {
                    self.take(8)?;
                    self.stack.push(Value::Float);
                }
                0x88 => self.stack.push(Value::Bool(true)),
                0x89 => self.stack.push(Value::Bool(false)),
                b'N' => self.stack.push(Value::None),
                b'X' => {
                    let len = self.u32()? as usize;
                    let v = self.string(len)?;
                    self.stack.push(v);
                }
                0x8c => {
                    let len = usize::from(self.u8()?);
                    let v = self.string(len)?;
                    self.stack.push(v);
                }
                b'c' => {
                    let module = self.line()?;
                    let name = self.line()?;
                    self.stack
                        .push(Value::Global(Global::resolve(module, name)?));
                }
                0x93 => {
                    let name = self.pop()?;
                    let module = self.pop()?;
                    let (Value::Str(module), Value::Str(name)) = (module, name) else {
                        return Err(format!("STACK_GLOBAL without names at byte {at}"));
                    };
                    self.stack
                        .push(Value::Global(Global::resolve(&module, &name)?));
                }
                b'R' => {
                    let args = self.pop()?;
                    let func = self.pop()?;
                    let (Value::Global(func), Value::Tuple(args)) = (&func, &args) else {
                        return Err(format!(
                            "REDUCE of a {} with a {} at byte {at}",
                            func.kind(),
                            args.kind()
                        ));
                    };
                    let v = reduce(*func, args).map_err(|e| format!("{e} (byte {at})"))?;
                    self.stack.push(v);
                }
                b'Q' => {
                    let pid = self.pop()?;
                    let v = self.persistent_load(&pid)?;
                    self.stack.push(v);
                }
                b's' => {
                    let v = self.pop()?;
                    let k = self.pop()?;
                    self.top_dict()?.borrow_mut().push((k, v));
                }
                b'u' => {
                    let items = self.pop_mark()?;
                    if items.len() % 2 != 0 {
                        return Err(format!("SETITEMS with an odd item count at byte {at}"));
                    }
                    let dict = self.top_dict()?;
                    let mut items = items.into_iter();
                    while let (Some(k), Some(v)) = (items.next(), items.next()) {
                        dict.borrow_mut().push((k, v));
                    }
                }
                b'a' => {
                    let v = self.pop()?;
                    self.top_list()?.borrow_mut().push(v);
                }
                b'e' => {
                    let items = self.pop_mark()?;
                    self.top_list()?.borrow_mut().extend(items);
                }
                b'b' => {
                    // An OrderedDict's attributes (its `_metadata` of module versions): unused.
                    let state = self.pop()?;
                    if !matches!(self.top()?, Value::Dict(_)) {
                        return Err(format!(
                            "BUILD of a {} with a {} at byte {at}",
                            self.top()?.kind(),
                            state.kind()
                        ));
                    }
                }
                _ => {
                    return Err(format!(
                        "unsupported pickle opcode {} (0x{op:02x}) at byte {at}",
                        opcode_name(op)
                    ));
                }
            }
        }
    }

    /// `("storage", <storage class>, key, location, numel)`; legacy files add a view field that
    /// is `None` in every file PyTorch >= 0.4 writes.
    fn persistent_load(&mut self, pid: &Value) -> PResult<Value> {
        let Value::Tuple(items) = pid else {
            return Err(format!("persistent id is a {}", pid.kind()));
        };
        let (kind, class, key, numel) = match &items[..] {
            [k, c, key, _location, n] | [k, c, key, _location, n, Value::None] => (k, c, key, n),
            _ => {
                return Err(format!(
                    "persistent id of {} fields is not supported",
                    items.len()
                ));
            }
        };
        let (Value::Str(kind), Value::Global(Global::StorageType(dtype)), Value::Str(key)) =
            (kind, class, key)
        else {
            return Err(format!(
                "persistent id ({}, {}, {}, ...) is not a storage",
                kind.kind(),
                class.kind(),
                key.kind()
            ));
        };
        if kind.as_ref() != "storage" {
            return Err(format!("persistent id of type {kind} is not supported"));
        }
        let numel = numel.as_usize("storage size")?;
        let storage = match self.storages.get(key.as_ref()) {
            Some(s) if s.dtype == *dtype && s.numel == numel => s.clone(),
            Some(s) => {
                return Err(format!(
                    "storage {key} is referenced as {numel} {dtype:?} and as {} {:?}",
                    s.numel, s.dtype
                ));
            }
            None => {
                let s = Rc::new(StorageRef {
                    dtype: *dtype,
                    key: key.to_string(),
                    numel,
                });
                self.storages.insert(key.to_string(), s.clone());
                s
            }
        };
        Ok(Value::Storage(storage))
    }
}

/// Applies an allowed callable to its arguments.
fn reduce(func: Global, args: &[Value]) -> PResult<Value> {
    match func {
        Global::OrderedDict => match args {
            [] => Ok(Value::new_dict()),
            [Value::List(l)] if l.borrow().is_empty() => Ok(Value::new_dict()),
            _ => Err("OrderedDict with arguments is not supported".to_string()),
        },
        // (storage, storage_offset, size, stride, requires_grad, backward_hooks[, metadata])
        Global::RebuildTensorV2 => {
            if !(6..=7).contains(&args.len()) {
                return Err(format!("_rebuild_tensor_v2 with {} arguments", args.len()));
            }
            let Value::Storage(storage) = &args[0] else {
                return Err(format!("_rebuild_tensor_v2 of a {}", args[0].kind()));
            };
            let offset = args[1].as_usize("storage offset")?;
            let shape = args[2].as_usizes("tensor size")?;
            let stride = args[3].as_usizes("tensor stride")?;
            if shape.len() != stride.len() {
                return Err(format!(
                    "tensor size {shape:?} and stride {stride:?} differ in rank"
                ));
            }
            check_view(storage, offset, &shape, &stride)?;
            // Metadata flags (such as conjugate or negative bits) would change the values.
            match args.get(6) {
                None | Some(Value::None) => {}
                Some(Value::Dict(m)) if m.borrow().is_empty() => {}
                Some(other) => {
                    return Err(format!(
                        "tensor metadata ({}) is not supported",
                        other.kind()
                    ));
                }
            }
            Ok(Value::Tensor(Rc::new(TensorRef {
                storage: storage.clone(),
                offset,
                shape,
                stride,
            })))
        }
        // (data, requires_grad, backward_hooks)
        Global::RebuildParameter => match args {
            [t @ Value::Tensor(_), _, _] => Ok(t.clone()),
            _ => Err("_rebuild_parameter without a tensor".to_string()),
        },
        Global::StorageType(dtype) => Err(format!("{dtype:?} storage class called directly")),
    }
}

/// Every element the view can reach must lie inside its storage. A view with more elements than
/// its storage (an expanded tensor, or a forged size) is refused before anything is allocated.
fn check_view(
    storage: &StorageRef,
    offset: usize,
    shape: &[usize],
    stride: &[usize],
) -> PResult<()> {
    let numel = shape.iter().try_fold(1usize, |acc, &n| acc.checked_mul(n));
    if numel.is_none_or(|n| n > storage.numel) {
        return Err(format!(
            "tensor of size {shape:?} has more elements than storage {} ({})",
            storage.key, storage.numel
        ));
    }
    if shape.contains(&0) {
        return Ok(());
    }
    let last = shape.iter().zip(stride).try_fold(offset, |acc, (&n, &s)| {
        (n - 1).checked_mul(s).and_then(|x| acc.checked_add(x))
    });
    match last {
        Some(last) if last < storage.numel => Ok(()),
        _ => Err(format!(
            "tensor view (offset {offset}, size {shape:?}, stride {stride:?}) exceeds storage {} of {} elements",
            storage.key, storage.numel
        )),
    }
}

fn opcode_name(op: u8) -> &'static str {
    match op {
        b'0' => "POP",
        b'1' => "POP_MARK",
        b'2' => "DUP",
        b'B' => "BINBYTES",
        b'C' => "SHORT_BINBYTES",
        b'F' => "FLOAT",
        b'I' => "INT",
        b'L' => "LONG",
        b'P' => "PERSID",
        b'S' => "STRING",
        b'T' => "BINSTRING",
        b'U' => "SHORT_BINSTRING",
        b'V' => "UNICODE",
        b'd' => "DICT",
        b'g' => "GET",
        b'i' => "INST",
        b'l' => "LIST",
        b'o' => "OBJ",
        b'p' => "PUT",
        0x81 => "NEWOBJ",
        0x82 => "EXT1",
        0x83 => "EXT2",
        0x84 => "EXT4",
        0x8b => "LONG4",
        0x8d => "BINUNICODE8",
        0x8e => "BINBYTES8",
        0x8f => "EMPTY_SET",
        0x90 => "ADDITEMS",
        0x91 => "FROZENSET",
        0x92 => "NEWOBJ_EX",
        0x96 => "BYTEARRAY8",
        0x97 => "NEXT_BUFFER",
        0x98 => "READONLY_BUFFER",
        _ => "unknown",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(pickle: &[u8]) -> PResult<Value> {
        Unpickler::new(pickle).load()
    }

    #[test]
    fn unknown_global_is_named() {
        let err = run(b"\x80\x02cos\nsystem\nq\x00X\x02\x00\x00\x00lsq\x01\x85R.").unwrap_err();
        assert!(err.contains("os.system"), "{err}");
    }

    #[test]
    fn unknown_opcode_is_named() {
        let err = run(b"\x80\x02(i__main__\nEvil\n.").unwrap_err();
        assert!(err.contains("INST") && err.contains("0x69"), "{err}");
    }

    #[test]
    fn view_outside_storage_is_rejected() {
        // _rebuild_tensor_v2(storage of 4 floats, offset 2, size (3,), stride (1,), False, {})
        let pickle = b"\x80\x02ctorch._utils\n_rebuild_tensor_v2\n(\
            (X\x07\x00\x00\x00storagectorch\nFloatStorage\nX\x01\x00\x00\x000X\x03\x00\x00\x00cpuK\x04tQ\
            K\x02K\x03\x85K\x01\x85\x89}tR.";
        let err = run(pickle).unwrap_err();
        assert!(err.contains("exceeds storage 0"), "{err}");
    }

    #[test]
    fn memoized_dict_sees_later_items() {
        // d = {}; memo[0] = d; d["a"] = 1; return (d, memo[0])
        let v = run(b"\x80\x02}q\x00X\x01\x00\x00\x00aK\x01sh\x00\x86.").unwrap();
        let Value::Tuple(items) = v else { panic!() };
        let Value::Dict(from_memo) = &items[1] else {
            panic!()
        };
        assert_eq!(from_memo.borrow().len(), 1);
    }
}
