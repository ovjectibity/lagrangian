use serde;
use typeshare::typeshare;
use utoipa::ToSchema;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, ToSchema)]
#[typeshare]
pub enum GeomType {
    Plane,
    Sphere,
    Cylinder,
    Capsule,
    Box,
    Mesh,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, ToSchema)]
#[typeshare]
pub struct Geom {
    pub name: Option<String>,
    pub pos: Option<Vec<f32>>,
    pub quat: Option<Vec<f32>>,
    pub euler: Option<Vec<f32>>,
    pub geom_t: Option<GeomType>,
    pub size: Vec<f32>,
    pub mesh_obj: Option<String>,
    pub group: Option<i32>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, ToSchema)]
#[typeshare]
pub struct Body {
    pub name: Option<String>,
    pub body_id: Option<i32>,
    pub pos: Option<Vec<f32>>,
    pub quat: Option<Vec<f32>>,
    pub euler: Option<Vec<f32>>,
    pub bodies: Option<Vec<Body>>,
    pub geoms: Option<Vec<Geom>>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, ToSchema)]
#[typeshare]
pub struct WorldBody {
    pub bodies: Vec<Body>,
}
