use rocket::serde::json::Json;
use serde::Serialize;

use crate::commondef::model::WorldBody;

#[derive(Serialize)]
pub struct DefaultModelResponse {
    pub id: String,
    pub model: WorldBody,
}

#[get("/models/default")]
pub fn get_default_model()
-> Json<DefaultModelResponse>
{
    Json(DefaultModelResponse {
        id: "default".to_string(),
        model: crate::simulator::get_test_model(),
    })
}
