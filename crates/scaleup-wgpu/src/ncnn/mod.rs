//! ncnn model files: `.param` is a text graph, `.bin` the weights.

pub mod bin;
pub mod graph;
pub mod param;

pub use graph::{
    Activation, BinOp, Graph, Layer, LayerKind, infer_channels, infer_scale, load_graph,
};
pub use param::{WeightSpec, parse_param};
