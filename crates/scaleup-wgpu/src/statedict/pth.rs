//! `.pth` checkpoints written by `torch.save`: a zip archive (PyTorch >= 1.6) or the legacy
//! sequence of pickles followed by the raw storages.
//!
//! The pickle is read by a restricted unpickler that only builds containers, scalars and tensor
//! references: nothing from the file is executed, and any global or opcode outside what weight
//! files use is an error naming it.

use super::{DType, TensorF32, decode};
use scaleup_core::{Result, ScaleUpError};
use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap, HashSet};
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

/// Highest tensor rank accepted (convolution weights have 4, 3D ones 5).
const MAX_RANK: usize = 8;

/// Ceiling for `data.pkl`: the largest real one seen is under 1 MiB.
const MAX_PICKLE_BYTES: u64 = 64 << 20;

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
    // Absent before PyTorch 1.10, where files were always little-endian.
    let byteorder = format!("{root}byteorder");
    if zip.index_for_name(&byteorder).is_some() {
        let order = read_entry(&mut zip, &byteorder, 16)?;
        if order != b"little" {
            return Err(format!(
                "byte order {} is not supported",
                shown(&String::from_utf8_lossy(&order))
            ));
        }
    }
    let pickle = read_entry(&mut zip, &pickle_name, MAX_PICKLE_BYTES)?;
    let root_value = Unpickler::new(&pickle).load()?;
    let tensors = state_dict_tensors(root_value)?;

    let mut storages: HashMap<String, Vec<u8>> = HashMap::new();
    for (_, t) in &tensors {
        let s = &t.storage;
        if !storages.contains_key(&s.key) {
            let expected = s.byte_len()?;
            let name = format!("{root}data/{}", s.key);
            let raw = read_entry(&mut zip, &name, expected as u64)?;
            if raw.len() != expected {
                return Err(format!(
                    "zip entry {} holds {} bytes, storage {} needs {expected}",
                    shown(&name),
                    raw.len(),
                    shown(&s.key)
                ));
            }
            storages.insert(s.key.clone(), raw);
        }
    }
    check_total(&tensors)?;
    let mut out = BTreeMap::new();
    for (key, t) in tensors {
        let t = materialize(&t, &storages[&t.storage.key])
            .map_err(|e| format!("tensor {}: {e}", shown(&key)))?;
        out.insert(key, t);
    }
    Ok(out)
}

/// Reads at most `limit` bytes of a stored entry. `torch.save` never compresses, so a compressed
/// entry (whose inflated size could be anything) is refused.
fn read_entry(
    zip: &mut zip::ZipArchive<Cursor<&[u8]>>,
    name: &str,
    limit: u64,
) -> PResult<Vec<u8>> {
    let entry = zip
        .by_name(name)
        .map_err(|e| format!("zip entry {}: {e}", shown(name)))?;
    if entry.compression() != zip::CompressionMethod::Stored {
        return Err(format!(
            "zip entry {} is compressed ({}); torch.save writes uncompressed entries",
            shown(name),
            entry.compression()
        ));
    }
    if entry.size() > limit {
        return Err(format!(
            "zip entry {} holds {} bytes, more than the {limit} expected",
            shown(name),
            entry.size()
        ));
    }
    let mut buf = Vec::new();
    entry
        .take(limit)
        .read_to_end(&mut buf)
        .map_err(|e| format!("zip entry {}: {e}", shown(name)))?;
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
        Ok((v, std::mem::take(&mut u.storages)))
    };
    if !matches!(next("magic number")?.0, Value::Int(LEGACY_MAGIC)) {
        return Err("not a PyTorch checkpoint (bad legacy magic number)".to_string());
    }
    match next("protocol version")?.0 {
        Value::Int(LEGACY_PROTOCOL) => {}
        Value::Int(v) => return Err(format!("legacy protocol version {v} is not supported")),
        other => {
            return Err(format!(
                "legacy protocol version is a {}, not an int",
                other.kind()
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
        let storage = storages.get(key.as_ref()).ok_or_else(|| {
            format!(
                "legacy storage {} is listed but never referenced",
                shown(key)
            )
        })?;
        let count = bytes
            .get(pos..pos + 8)
            .ok_or_else(|| format!("legacy storage {}: truncated header", shown(key)))?;
        let count = i64::from_le_bytes(count.try_into().unwrap());
        if usize::try_from(count).ok() != Some(storage.numel) {
            return Err(format!(
                "legacy storage {} holds {count} elements, its tensors expect {}",
                shown(key),
                storage.numel
            ));
        }
        pos += 8;
        let end = storage
            .byte_len()?
            .checked_add(pos)
            .filter(|&end| end <= bytes.len())
            .ok_or_else(|| format!("legacy storage {}: truncated data", shown(key)))?;
        raw.insert(key.to_string(), &bytes[pos..end]);
        pos = end;
    }

    let tensors = state_dict_tensors(root_value)?;
    for (key, t) in &tensors {
        if !raw.contains_key(&t.storage.key) {
            return Err(format!(
                "tensor {}: storage {} is not in the file",
                shown(key),
                shown(&t.storage.key)
            ));
        }
    }
    check_total(&tensors)?;
    let mut out = BTreeMap::new();
    for (key, t) in tensors {
        let t = materialize(&t, raw[&t.storage.key])
            .map_err(|e| format!("tensor {}: {e}", shown(&key)))?;
        out.insert(key, t);
    }
    Ok(out)
}

/// Many names may view one storage. Their storages having been read from the file, the
/// elements materialised in all may not exceed twice the elements those storages hold (plus a
/// little), so a forged file cannot turn one storage into an unbounded allocation.
fn check_total(tensors: &[(String, Rc<TensorRef>)]) -> PResult<()> {
    let mut seen = HashSet::new();
    let stored = tensors
        .iter()
        .filter(|(_, t)| seen.insert(t.storage.key.as_str()))
        .fold(0usize, |acc, (_, t)| acc.saturating_add(t.storage.numel));
    let cap = stored.saturating_mul(2).saturating_add(1024);
    let mut total = 0usize;
    for (key, t) in tensors {
        total = total.saturating_add(t.shape.iter().product());
        if total > cap {
            return Err(format!(
                "tensor {}: the tensors would hold more than twice the {stored} elements stored in the file",
                shown(key)
            ));
        }
    }
    Ok(())
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
            shown(&s.key),
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
            .ok_or_else(|| format!("storage {} is too large", shown(&self.key)))
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
            _ => {
                return Err(format!(
                    "unsupported pickle global {}.{}",
                    shown(module),
                    shown(name)
                ));
            }
        })
    }
}

/// File text quoted in an error: at most 100 characters, control characters escaped.
fn shown(text: &str) -> String {
    let mut out = String::new();
    for (i, c) in text.chars().enumerate() {
        if i == 100 {
            out.push_str("...");
            break;
        }
        if c.is_control() {
            out.extend(c.escape_default());
        } else {
            out.push(c);
        }
    }
    out
}

/// The elements of a tuple or list. Dropping them is iterative: a file can nest values a
/// million deep (or make a list contain itself), and a recursive drop would overflow the stack.
#[derive(Default)]
struct Items(Vec<Value>);

/// The entries of a dict, dropped iteratively like [`Items`].
#[derive(Default)]
struct Pairs(Vec<(Value, Value)>);

impl Drop for Items {
    fn drop(&mut self) {
        release(std::mem::take(&mut self.0));
    }
}

impl Drop for Pairs {
    fn drop(&mut self) {
        release(self.0.drain(..).flat_map(|(k, v)| [k, v]).collect());
    }
}

/// Drops `values` without recursion: the children of a container this was the last owner of
/// are moved onto the work list before the emptied container is dropped.
fn release(mut pending: Vec<Value>) {
    while let Some(v) = pending.pop() {
        match v {
            Value::Tuple(rc) => {
                if let Ok(mut items) = Rc::try_unwrap(rc) {
                    pending.append(&mut items.0);
                }
            }
            Value::List(rc) => {
                if let Ok(cell) = Rc::try_unwrap(rc) {
                    pending.append(&mut cell.into_inner().0);
                }
            }
            Value::Dict(rc) => {
                if let Ok(cell) = Rc::try_unwrap(rc) {
                    pending.extend(cell.into_inner().0.drain(..).flat_map(|(k, v)| [k, v]));
                }
            }
            _ => {}
        }
    }
}

impl std::ops::Deref for Items {
    type Target = Vec<Value>;
    fn deref(&self) -> &Vec<Value> {
        &self.0
    }
}

impl std::ops::DerefMut for Items {
    fn deref_mut(&mut self) -> &mut Vec<Value> {
        &mut self.0
    }
}

impl std::ops::Deref for Pairs {
    type Target = Vec<(Value, Value)>;
    fn deref(&self) -> &Vec<(Value, Value)> {
        &self.0
    }
}

impl std::ops::DerefMut for Pairs {
    fn deref_mut(&mut self) -> &mut Vec<(Value, Value)> {
        &mut self.0
    }
}

type ListRef = Rc<RefCell<Items>>;
type DictRef = Rc<RefCell<Pairs>>;

/// Dicts and lists are shared so that a memoized container sees later `SETITEM`s, as in Python.
/// No `Debug`: values come from the file, may be cyclic, and are described with [`Value::kind`].
#[derive(Clone)]
enum Value {
    None,
    Bool(bool),
    Int(i128),
    /// Only options and metadata are floats; their value is never needed.
    Float,
    Str(Rc<str>),
    Tuple(Rc<Items>),
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
        Value::Dict(Rc::default())
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
                b']' => self.stack.push(Value::List(Rc::default())),
                b')' => self.stack.push(Value::Tuple(Rc::default())),
                b't' => {
                    let items = self.pop_mark()?;
                    self.stack.push(Value::Tuple(Rc::new(Items(items))));
                }
                0x85..=0x87 => {
                    let n = usize::from(op - 0x84);
                    let mut items = Vec::with_capacity(n);
                    for _ in 0..n {
                        items.push(self.pop()?);
                    }
                    items.reverse();
                    self.stack.push(Value::Tuple(Rc::new(Items(items))));
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
            return Err(format!(
                "persistent id of type {} is not supported",
                shown(kind)
            ));
        }
        let numel = numel.as_usize("storage size")?;
        let storage = match self.storages.get(key.as_ref()) {
            Some(s) if s.dtype == *dtype && s.numel == numel => s.clone(),
            Some(s) => {
                return Err(format!(
                    "storage {} is referenced as {numel} {dtype:?} and as {} {:?}",
                    shown(key),
                    s.numel,
                    s.dtype
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
            // Sizes and strides are printed in errors, so their length is bounded first.
            if shape.len() > MAX_RANK || shape.len() != stride.len() {
                return Err(format!(
                    "tensor size of rank {} with stride of rank {} is not supported",
                    shape.len(),
                    stride.len()
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
            shown(&storage.key),
            storage.numel
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
            shown(&storage.key),
            storage.numel
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

    fn run_err(pickle: &[u8]) -> String {
        match run(pickle) {
            Ok(v) => panic!("the pickle loaded a {}", v.kind()),
            Err(e) => e,
        }
    }

    #[test]
    fn unknown_global_is_named() {
        let err = run_err(b"\x80\x02cos\nsystem\nq\x00X\x02\x00\x00\x00lsq\x01\x85R.");
        assert!(err.contains("os.system"), "{err}");
    }

    #[test]
    fn unknown_opcode_is_named() {
        let err = run_err(b"\x80\x02(i__main__\nEvil\n.");
        assert!(err.contains("INST") && err.contains("0x69"), "{err}");
    }

    #[test]
    fn view_outside_storage_is_rejected() {
        // _rebuild_tensor_v2(storage of 4 floats, offset 2, size (3,), stride (1,), False, {})
        let pickle = b"\x80\x02ctorch._utils\n_rebuild_tensor_v2\n(\
            (X\x07\x00\x00\x00storagectorch\nFloatStorage\nX\x01\x00\x00\x000X\x03\x00\x00\x00cpuK\x04tQ\
            K\x02K\x03\x85K\x01\x85\x89}tR.";
        let err = run_err(pickle);
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

    fn stored_zip(entries: &[(&str, &[u8])]) -> Vec<u8> {
        use std::io::Write;
        let mut w = zip::ZipWriter::new(Cursor::new(Vec::new()));
        let opts = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Stored);
        for (name, data) in entries {
            w.start_file(*name, opts).unwrap();
            w.write_all(data).unwrap();
        }
        w.finish().unwrap().into_inner()
    }

    fn push_str(p: &mut Vec<u8>, s: &str) {
        p.push(b'X');
        p.extend((s.len() as u32).to_le_bytes());
        p.extend(s.as_bytes());
    }

    /// `{"k0": t, "k1": t, ...}` where `t` is a 4-float tensor on storage "0", memoized once.
    fn shared_tensor_pickle(keys: usize) -> Vec<u8> {
        let mut p = b"\x80\x02}(".to_vec();
        for i in 0..keys {
            push_str(&mut p, &format!("k{i}"));
            if i == 0 {
                p.extend_from_slice(b"ctorch._utils\n_rebuild_tensor_v2\n((");
                push_str(&mut p, "storage");
                p.extend_from_slice(b"ctorch\nFloatStorage\n");
                push_str(&mut p, "0");
                push_str(&mut p, "cpu");
                p.extend_from_slice(b"K\x04tQK\x00K\x04\x85K\x01\x85\x89}tRq\x01");
            } else {
                p.extend_from_slice(b"h\x01");
            }
        }
        p.extend_from_slice(b"u.");
        p
    }

    fn nested_tuples(depth: usize) -> Vec<u8> {
        let mut p = b"\x80\x02K\x00".to_vec();
        p.resize(p.len() + depth, 0x85);
        p.push(b'.');
        p
    }

    fn legacy_fixture() -> Vec<u8> {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/fixtures/sd/tiny_compact_legacy.pth"
        );
        std::fs::read(path).unwrap()
    }

    #[test]
    fn shared_storage_zip_loads() {
        let data: Vec<u8> = (0..4).flat_map(|i| (i as f32).to_le_bytes()).collect();
        let zip = stored_zip(&[
            ("a/data.pkl", &shared_tensor_pickle(2)),
            ("a/data/0", &data),
        ]);
        let sd = load_zip(&zip).unwrap();
        assert_eq!(sd.len(), 2);
        assert_eq!(sd["k1"].data, vec![0.0_f32, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn deep_nesting_is_an_error_not_a_crash() {
        let pickle = nested_tuples(1_000_000);
        let err = load_zip(&stored_zip(&[("a/data.pkl", &pickle)])).unwrap_err();
        assert!(err.contains("holds a tuple"), "{err}");
        let err = load_legacy(&pickle).unwrap_err();
        assert!(err.contains("magic"), "{err}");
        // Without STOP the nested value is dropped on the error path.
        let err = load_legacy(&pickle[..pickle.len() - 1]).unwrap_err();
        assert!(err.contains("truncated"), "{err}");
    }

    #[test]
    fn self_referential_list_is_an_error() {
        let pickle = b"\x80\x02]q\x00h\x00a.";
        let err = load_zip(&stored_zip(&[("a/data.pkl", pickle)])).unwrap_err();
        assert!(err.contains("holds a list"), "{err}");
        assert!(load_legacy(pickle).unwrap_err().contains("magic"));
    }

    #[test]
    fn shared_storage_amplification_is_refused() {
        let data = [0u8; 16];
        let zip = stored_zip(&[
            ("a/data.pkl", &shared_tensor_pickle(1000)),
            ("a/data/0", &data),
        ]);
        let err = load_zip(&zip).unwrap_err();
        assert!(err.contains("tensor k") && err.contains("twice"), "{err}");
    }

    #[test]
    fn storage_entry_of_wrong_size_is_refused() {
        let pickle = shared_tensor_pickle(1);
        let short = stored_zip(&[("a/data.pkl", &pickle), ("a/data/0", &[0u8; 12])]);
        let err = load_zip(&short).unwrap_err();
        assert!(err.contains("a/data/0 holds 12 bytes"), "{err}");
        let long = stored_zip(&[("a/data.pkl", &pickle), ("a/data/0", &[0u8; 20])]);
        let err = load_zip(&long).unwrap_err();
        assert!(err.contains("more than the 16 expected"), "{err}");
    }

    #[test]
    fn entry_reads_are_bounded() {
        let zip = stored_zip(&[("a/x", &[0u8; 32])]);
        let mut archive = zip::ZipArchive::new(Cursor::new(&zip[..])).unwrap();
        let err = read_entry(&mut archive, "a/x", 16).unwrap_err();
        assert!(err.contains("more than the 16 expected"), "{err}");
        assert_eq!(read_entry(&mut archive, "a/x", 32).unwrap().len(), 32);
    }

    #[test]
    fn compressed_entry_is_refused() {
        let mut zip = stored_zip(&[("a/byteorder", b"little"), ("a/data.pkl", b"\x80\x02}.")]);
        // Mark the first entry as deflated in its local header and its central directory record.
        assert_eq!(&zip[..4], b"PK\x03\x04");
        zip[8] = 8;
        let central = zip.windows(4).position(|w| w == b"PK\x01\x02").unwrap();
        zip[central + 10] = 8;
        let err = load_zip(&zip).unwrap_err();
        assert!(err.contains("a/byteorder"), "{err}");
    }

    #[test]
    fn foreign_byte_order_is_refused() {
        let zip = stored_zip(&[("a/byteorder", b"big"), ("a/data.pkl", b"\x80\x02}.")]);
        assert!(load_zip(&zip).unwrap_err().contains("byte order big"));
    }

    #[test]
    fn hostile_pickles_are_errors() {
        let cases: [(&[u8], &str); 4] = [
            (b"\x80\x02X\x05\x00\x00\x00ab", "truncated"),
            (b"\x80\x02h\x05.", "memo entry 5"),
            (b"\x80\x02K\x01(\x85.", "MARK"),
            (b"\x80\x02}(K\x01u.", "odd"),
        ];
        for (pickle, expected) in cases {
            let err = run_err(pickle);
            assert!(err.contains(expected), "{expected}: {err}");
        }
    }

    #[test]
    fn global_names_in_errors_are_bounded_and_escaped() {
        let mut pickle = b"\x80\x02c".to_vec();
        pickle.extend(std::iter::repeat_n(b'a', 1000));
        pickle.extend_from_slice(b"\nb\x01\n.");
        let err = run_err(&pickle);
        assert!(err.len() < 300, "{err}");
        assert!(err.contains("...") && err.contains("b\\u{1}"), "{err}");
    }

    #[test]
    fn truncated_legacy_storage_is_an_error() {
        let bytes = legacy_fixture();
        let err = load_legacy(&bytes[..bytes.len() - 3]).unwrap_err();
        assert!(err.contains("truncated data"), "{err}");
    }

    #[test]
    fn legacy_element_count_mismatch_is_an_error() {
        let mut bytes = legacy_fixture();
        let mut pos = 0;
        for _ in 0..5 {
            let mut u = Unpickler::new(&bytes[pos..]);
            u.load().unwrap();
            pos += u.pos;
        }
        bytes[pos] ^= 1;
        let err = load_legacy(&bytes).unwrap_err();
        assert!(err.contains("elements, its tensors expect"), "{err}");
    }
}
