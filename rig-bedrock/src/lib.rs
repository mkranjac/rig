pub mod client;
pub mod completion;
pub mod embedding;
pub mod types;

use aws_smithy_types::{Document, Number};
use serde_json::{Map, Value};
use std::collections::HashMap;

/// Convert a `aws_smithy_types::Document` into `serde_json::Value`
pub fn document_to_json(doc: Document) -> Value {
    match doc {
        Document::Object(obj) => {
            let documents = obj
                .into_iter()
                .map(|(k, v)| (k, document_to_json(v)))
                .collect::<Map<_, _>>();
            Value::Object(documents)
        }
        Document::Array(arr) => {
            let documents = arr.into_iter().map(document_to_json).collect();
            Value::Array(documents)
        }
        Document::Number(Number::PosInt(number)) => Value::Number(serde_json::Number::from(number)),
        Document::Number(Number::NegInt(number)) => Value::Number(serde_json::Number::from(number)),
        Document::Number(Number::Float(number)) => match serde_json::Number::from_f64(number) {
            Some(n) => Value::Number(n),
            // https://www.rfc-editor.org/rfc/rfc7159
            // Numeric values that cannot be represented in the grammar (such as Infinity and NaN) are not permitted.
            None => Value::Null,
        },
        Document::String(s) => Value::String(s),
        Document::Bool(b) => Value::Bool(b),
        Document::Null => Value::Null,
    }
}

/// Convert a `serde_json::Value` into `aws_smithy_types::Document`
pub fn json_to_document(value: Value) -> Document {
    match value {
        Value::Null => Document::Null,
        Value::Bool(b) => Document::Bool(b),
        Value::Number(num) => {
            if let Some(i) = num.as_i64() {
                match i > 0 {
                    true => Document::Number(Number::PosInt(i as u64)),
                    false => Document::Number(Number::NegInt(i)),
                }
            } else if let Some(f) = num.as_f64() {
                Document::Number(Number::Float(f))
            } else {
                Document::Null
            }
        }
        Value::String(s) => Document::String(s),
        Value::Array(arr) => {
            let documents = arr.into_iter().map(json_to_document).collect();
            Document::Array(documents)
        }
        Value::Object(obj) => {
            let documents = obj
                .into_iter()
                .map(|(k, v)| (k, json_to_document(v)))
                .collect::<HashMap<_, _>>();
            Document::Object(documents)
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::document_to_json;
    use super::json_to_document;
    use aws_smithy_types::Document;
    use serde_json::Value;

    #[test]
    fn test_json_to_document() {
        let json = r#"
            {
                "type": "object",
                "properties": {
                    "x": {
                        "type": "number",
                        "description": "The first number to add"
                    },
                    "y": {
                        "type": "number",
                        "description": "The second number to add"
                    }
                },
                "required":["x", "y"]
            }
        "#;

        let value: Value = serde_json::from_str(json).unwrap();
        let document = json_to_document(value);
        println!("{:?}", document);
    }

    #[test]
    fn test_document_to_json() {
        let document = Document::Object(HashMap::from([
            (
                String::from("type"),
                Document::String(String::from("object")),
            ),
            (
                String::from("properties"),
                Document::Object(HashMap::from([
                    (
                        String::from("x"),
                        Document::Object(HashMap::from([
                            (
                                String::from("type"),
                                Document::String(String::from("number")),
                            ),
                            (
                                String::from("description"),
                                Document::String(String::from("The first number to add")),
                            ),
                        ])),
                    ),
                    (
                        String::from("y"),
                        Document::Object(HashMap::from([
                            (
                                String::from("type"),
                                Document::String(String::from("number")),
                            ),
                            (
                                String::from("description"),
                                Document::String(String::from("The second number to add")),
                            ),
                        ])),
                    ),
                ])),
            ),
            (
                String::from("required"),
                Document::Array(vec![
                    Document::String(String::from("x")),
                    Document::String(String::from("y")),
                ]),
            ),
        ]));

        let json = document_to_json(document);
        println!("{:?}", json);
    }
}
