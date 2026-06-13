use lagrangian::commondef::record::TrainingGraphRecord;
use serde_json::Value;

fn main() {
    let schema = schemars::schema_for!(TrainingGraphRecord);
    let mut value: Value = serde_json::to_value(&schema).expect("failed to serialize schema");
    // schemars 1.x emits $defs (Draft 2020-12); MongoDB expects definitions (Draft 4/7)
    rewrite_defs(&mut value);
    println!("{}", serde_json::to_string_pretty(&value).expect("failed to pretty-print schema"));
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
