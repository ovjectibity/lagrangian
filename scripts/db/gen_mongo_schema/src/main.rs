use lagrangian::commondef::record::{
    LogStream, MetricsStream, TestGraphRecord, TrainingGraphRecord,
};
use serde_json::Value;

fn main() {
    let schema_name = std::env::args()
        .nth(1)
        .expect("Usage: gen_mongo_schema <training_graph|log_stream|metrics_stream|test_graph>");

    let value = match schema_name.as_str() {
        "training_graph" => {
            let s = schemars::schema_for!(TrainingGraphRecord);
            serde_json::to_value(&s).expect("serialize")
        }
        "log_stream" => {
            let s = schemars::schema_for!(LogStream);
            serde_json::to_value(&s).expect("serialize")
        }
        "metrics_stream" => {
            let s = schemars::schema_for!(MetricsStream);
            serde_json::to_value(&s).expect("serialize")
        }
        "test_graph" => {
            let s = schemars::schema_for!(TestGraphRecord);
            serde_json::to_value(&s).expect("serialize")
        }
        other => panic!("Unknown schema: {other}"),
    };

    let mut value = value;
    rewrite_defs(&mut value);
    println!("{}", serde_json::to_string_pretty(&value).expect("pretty-print"));
}

fn rewrite_defs(v: &mut Value) {
    match v {
        Value::Object(map) => {
            if let Some(defs) = map.remove("$defs") {
                map.insert("definitions".to_string(), defs);
            }
            for val in map.values_mut() {
                rewrite_defs(val);
            }
        }
        Value::Array(arr) => {
            for val in arr.iter_mut() {
                rewrite_defs(val);
            }
        }
        Value::String(s) => {
            if s.starts_with("#/$defs/") {
                *s = s.replacen("#/$defs/", "#/definitions/", 1);
            }
        }
        _ => {}
    }
}
