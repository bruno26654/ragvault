//! Metadata filter DSL.
//!
//! Filters are parsed from the JSON shape used by the Python API:
//!
//! ```json
//! {"source": "manual"}
//! {"year": {"gte": 2024}}
//! {"$and": [{"tenant_id": "acme"}, {"status": {"ne": "draft"}}]}
//! ```
//!
//! Semantics (documented and tested):
//! - A missing field or `null` value fails every comparison except
//!   `exists: false` and `ne` (a missing field is "not equal" to any value).
//! - Type-incompatible comparisons (e.g. `gt` between string and number)
//!   evaluate to `false` rather than erroring, so a single odd document
//!   cannot poison a query.
//! - `contains` on a list checks membership; on a string checks substring.
//! - `NaN` never compares equal, greater or lower — all comparisons false.

use serde_json::Value;

use crate::{Error, Result};

const MAX_FILTER_DEPTH: usize = 32;

#[derive(Debug, Clone, PartialEq)]
pub enum Filter {
    And(Vec<Filter>),
    Or(Vec<Filter>),
    Not(Box<Filter>),
    Cmp { field: String, op: CmpOp },
    True,
}

#[derive(Debug, Clone, PartialEq)]
pub enum CmpOp {
    Eq(Value),
    Ne(Value),
    In(Vec<Value>),
    NotIn(Vec<Value>),
    Gt(Value),
    Gte(Value),
    Lt(Value),
    Lte(Value),
    Contains(Value),
    ContainsAny(Vec<Value>),
    ContainsAll(Vec<Value>),
    Exists(bool),
    Prefix(String),
}

impl Filter {
    /// Parse the JSON filter DSL. `None`/`null` parses to `Filter::True`.
    pub fn parse(value: &Value) -> Result<Filter> {
        Self::parse_depth(value, 0)
    }

    fn parse_depth(value: &Value, depth: usize) -> Result<Filter> {
        if depth > MAX_FILTER_DEPTH {
            return Err(Error::InvalidFilter(format!(
                "filter nesting deeper than {MAX_FILTER_DEPTH} levels"
            )));
        }
        match value {
            Value::Null => Ok(Filter::True),
            Value::Object(map) => {
                if map.is_empty() {
                    return Ok(Filter::True);
                }
                let mut clauses = Vec::with_capacity(map.len());
                for (key, val) in map {
                    match key.as_str() {
                        "$and" | "and" => {
                            let items = as_array(val, key)?;
                            let mut sub = Vec::with_capacity(items.len());
                            for item in items {
                                sub.push(Self::parse_depth(item, depth + 1)?);
                            }
                            clauses.push(Filter::And(sub));
                        }
                        "$or" | "or" => {
                            let items = as_array(val, key)?;
                            let mut sub = Vec::with_capacity(items.len());
                            for item in items {
                                sub.push(Self::parse_depth(item, depth + 1)?);
                            }
                            clauses.push(Filter::Or(sub));
                        }
                        "$not" | "not" => {
                            clauses.push(Filter::Not(Box::new(Self::parse_depth(val, depth + 1)?)));
                        }
                        field => {
                            clauses.push(parse_field(field, val, depth)?);
                        }
                    }
                }
                if clauses.len() == 1 {
                    Ok(clauses.pop().expect("len checked"))
                } else {
                    Ok(Filter::And(clauses))
                }
            }
            other => Err(Error::InvalidFilter(format!(
                "filter must be an object or null, got: {other}"
            ))),
        }
    }

    /// Evaluate the filter against a metadata object.
    pub fn matches(&self, metadata: &Value) -> bool {
        match self {
            Filter::True => true,
            Filter::And(fs) => fs.iter().all(|f| f.matches(metadata)),
            Filter::Or(fs) => fs.iter().any(|f| f.matches(metadata)),
            Filter::Not(f) => !f.matches(metadata),
            Filter::Cmp { field, op } => {
                let value = lookup(metadata, field);
                eval_cmp(op, value)
            }
        }
    }

    pub fn is_true(&self) -> bool {
        matches!(self, Filter::True)
    }
}

fn as_array<'a>(val: &'a Value, key: &str) -> Result<&'a Vec<Value>> {
    val.as_array()
        .ok_or_else(|| Error::InvalidFilter(format!("operator {key} expects an array, got: {val}")))
}

fn parse_field(field: &str, val: &Value, _depth: usize) -> Result<Filter> {
    let op = match val {
        // A non-empty object value is always an operator map. Unknown keys
        // are hard errors (a typo must not silently become a literal
        // equality that matches nothing); nested equality uses dotted paths.
        Value::Object(ops) if !ops.is_empty() => {
            let mut sub = Vec::with_capacity(ops.len());
            for (op_key, op_val) in ops {
                if !is_op_key(op_key) {
                    return Err(Error::InvalidFilter(format!(
                        "unknown operator {op_key:?} for field {field:?}; \
                         use a dotted path like \"{field}.{op_key}\" for nested equality"
                    )));
                }
                sub.push(Filter::Cmp {
                    field: field.to_string(),
                    op: parse_op(op_key, op_val)?,
                });
            }
            if sub.len() == 1 {
                return Ok(sub.pop().expect("len checked"));
            }
            return Ok(Filter::And(sub));
        }
        other => CmpOp::Eq(other.clone()),
    };
    Ok(Filter::Cmp {
        field: field.to_string(),
        op,
    })
}

fn is_op_key(k: &str) -> bool {
    matches!(
        k.trim_start_matches('$'),
        "eq" | "ne"
            | "in"
            | "not_in"
            | "nin"
            | "gt"
            | "gte"
            | "lt"
            | "lte"
            | "contains"
            | "contains_any"
            | "contains_all"
            | "exists"
            | "prefix"
    )
}

fn parse_op(key: &str, val: &Value) -> Result<CmpOp> {
    let op = match key.trim_start_matches('$') {
        "eq" => CmpOp::Eq(val.clone()),
        "ne" => CmpOp::Ne(val.clone()),
        "in" => CmpOp::In(as_array(val, key)?.clone()),
        "not_in" | "nin" => CmpOp::NotIn(as_array(val, key)?.clone()),
        "gt" => CmpOp::Gt(val.clone()),
        "gte" => CmpOp::Gte(val.clone()),
        "lt" => CmpOp::Lt(val.clone()),
        "lte" => CmpOp::Lte(val.clone()),
        "contains" => CmpOp::Contains(val.clone()),
        "contains_any" => CmpOp::ContainsAny(as_array(val, key)?.clone()),
        "contains_all" => CmpOp::ContainsAll(as_array(val, key)?.clone()),
        "exists" => CmpOp::Exists(val.as_bool().ok_or_else(|| {
            Error::InvalidFilter(format!("exists expects a boolean, got: {val}"))
        })?),
        "prefix" => CmpOp::Prefix(
            val.as_str()
                .ok_or_else(|| {
                    Error::InvalidFilter(format!("prefix expects a string, got: {val}"))
                })?
                .to_string(),
        ),
        other => {
            return Err(Error::InvalidFilter(format!("unknown operator: {other}")));
        }
    };
    Ok(op)
}

/// Dotted-path lookup into a metadata object.
fn lookup<'a>(metadata: &'a Value, field: &str) -> Option<&'a Value> {
    let mut current = metadata;
    for part in field.split('.') {
        current = current.as_object()?.get(part)?;
    }
    Some(current)
}

fn eval_cmp(op: &CmpOp, value: Option<&Value>) -> bool {
    match op {
        CmpOp::Exists(want) => {
            let exists = matches!(value, Some(v) if !v.is_null());
            exists == *want
        }
        CmpOp::Ne(target) => match value {
            None | Some(Value::Null) => !target.is_null(),
            Some(v) => !json_eq(v, target),
        },
        CmpOp::NotIn(targets) => match value {
            None | Some(Value::Null) => true,
            Some(v) => !targets.iter().any(|t| json_eq(v, t)),
        },
        _ => {
            let Some(v) = value else { return false };
            if v.is_null() {
                return false;
            }
            match op {
                CmpOp::Eq(t) => json_eq(v, t),
                CmpOp::In(ts) => ts.iter().any(|t| json_eq(v, t)),
                CmpOp::Gt(t) => json_cmp(v, t).map(|o| o.is_gt()).unwrap_or(false),
                CmpOp::Gte(t) => json_cmp(v, t).map(|o| o.is_ge()).unwrap_or(false),
                CmpOp::Lt(t) => json_cmp(v, t).map(|o| o.is_lt()).unwrap_or(false),
                CmpOp::Lte(t) => json_cmp(v, t).map(|o| o.is_le()).unwrap_or(false),
                CmpOp::Contains(t) => contains(v, t),
                CmpOp::ContainsAny(ts) => ts.iter().any(|t| contains(v, t)),
                CmpOp::ContainsAll(ts) => ts.iter().all(|t| contains(v, t)),
                CmpOp::Prefix(p) => v
                    .as_str()
                    .map(|s| s.starts_with(p.as_str()))
                    .unwrap_or(false),
                CmpOp::Exists(_) | CmpOp::Ne(_) | CmpOp::NotIn(_) => unreachable!(),
            }
        }
    }
}

fn contains(value: &Value, target: &Value) -> bool {
    match value {
        Value::Array(items) => items.iter().any(|i| json_eq(i, target)),
        Value::String(s) => target.as_str().map(|t| s.contains(t)).unwrap_or(false),
        _ => false,
    }
}

/// Equality with numeric coercion (1 == 1.0) and NaN-never-equal semantics.
fn json_eq(a: &Value, b: &Value) -> bool {
    match (a.as_f64(), b.as_f64()) {
        (Some(x), Some(y)) => x == y, // NaN != NaN by IEEE semantics
        _ => a == b,
    }
}

/// Ordering for numbers and strings; `None` for incompatible types or NaN.
fn json_cmp(a: &Value, b: &Value) -> Option<std::cmp::Ordering> {
    if let (Some(x), Some(y)) = (a.as_f64(), b.as_f64()) {
        return x.partial_cmp(&y);
    }
    if let (Some(x), Some(y)) = (a.as_str(), b.as_str()) {
        return Some(x.cmp(y));
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn matches(filter: Value, metadata: Value) -> bool {
        Filter::parse(&filter).unwrap().matches(&metadata)
    }

    #[test]
    fn implicit_eq() {
        assert!(matches(
            json!({"source": "manual"}),
            json!({"source": "manual"})
        ));
        assert!(!matches(
            json!({"source": "manual"}),
            json!({"source": "web"})
        ));
        assert!(!matches(json!({"source": "manual"}), json!({})));
    }

    #[test]
    fn range_ops() {
        assert!(matches(
            json!({"year": {"gte": 2024}}),
            json!({"year": 2024})
        ));
        assert!(matches(
            json!({"year": {"gte": 2024}}),
            json!({"year": 2025.5})
        ));
        assert!(!matches(
            json!({"year": {"gte": 2024}}),
            json!({"year": 2023})
        ));
        // combined range on one field
        assert!(matches(
            json!({"year": {"gte": 2020, "lt": 2024}}),
            json!({"year": 2022})
        ));
        assert!(!matches(
            json!({"year": {"gte": 2020, "lt": 2024}}),
            json!({"year": 2024})
        ));
        // incompatible types never match
        assert!(!matches(
            json!({"year": {"gte": 2024}}),
            json!({"year": "old"})
        ));
    }

    #[test]
    fn missing_and_null_semantics() {
        assert!(!matches(json!({"a": {"gt": 1}}), json!({})));
        assert!(!matches(json!({"a": {"gt": 1}}), json!({"a": null})));
        assert!(matches(json!({"a": {"exists": false}}), json!({})));
        assert!(matches(json!({"a": {"exists": false}}), json!({"a": null})));
        assert!(matches(json!({"a": {"exists": true}}), json!({"a": 0})));
        // ne matches missing fields
        assert!(matches(json!({"a": {"ne": "x"}}), json!({})));
        assert!(!matches(json!({"a": {"ne": "x"}}), json!({"a": "x"})));
    }

    #[test]
    fn boolean_composition() {
        let f = json!({
            "$and": [
                {"tenant_id": "acme"},
                {"status": {"ne": "draft"}},
            ]
        });
        assert!(matches(
            f.clone(),
            json!({"tenant_id": "acme", "status": "final"})
        ));
        assert!(!matches(
            f.clone(),
            json!({"tenant_id": "acme", "status": "draft"})
        ));
        assert!(!matches(
            f,
            json!({"tenant_id": "other", "status": "final"})
        ));

        let f = json!({"$or": [{"a": 1}, {"b": 2}]});
        assert!(matches(f.clone(), json!({"a": 1})));
        assert!(matches(f.clone(), json!({"b": 2})));
        assert!(!matches(f, json!({"a": 2, "b": 1})));

        let f = json!({"$not": {"a": 1}});
        assert!(matches(f.clone(), json!({"a": 2})));
        assert!(!matches(f, json!({"a": 1})));
    }

    #[test]
    fn list_and_string_contains() {
        assert!(matches(
            json!({"tags": {"contains": "legal"}}),
            json!({"tags": ["legal", "hr"]})
        ));
        assert!(matches(
            json!({"body": {"contains": "cancel"}}),
            json!({"body": "cancellation rules"})
        ));
        assert!(matches(
            json!({"tags": {"contains_any": ["x", "hr"]}}),
            json!({"tags": ["legal", "hr"]})
        ));
        assert!(!matches(
            json!({"tags": {"contains_all": ["x", "hr"]}}),
            json!({"tags": ["legal", "hr"]})
        ));
        assert!(matches(
            json!({"path": {"prefix": "docs/"}}),
            json!({"path": "docs/a.md"})
        ));
    }

    #[test]
    fn in_and_not_in() {
        assert!(matches(json!({"k": {"in": [1, 2]}}), json!({"k": 2})));
        assert!(!matches(json!({"k": {"in": [1, 2]}}), json!({"k": 3})));
        assert!(matches(json!({"k": {"not_in": [1, 2]}}), json!({"k": 3})));
        assert!(matches(json!({"k": {"not_in": [1, 2]}}), json!({})));
    }

    #[test]
    fn dotted_paths_and_numeric_coercion() {
        assert!(matches(json!({"a.b": 1}), json!({"a": {"b": 1.0}})));
        assert!(!matches(json!({"a.b": 1}), json!({"a": 7})));
    }

    #[test]
    fn depth_limit() {
        let mut f = json!({"a": 1});
        for _ in 0..40 {
            f = json!({ "$not": f });
        }
        assert!(Filter::parse(&f).is_err());
    }

    #[test]
    fn rejects_bad_shapes() {
        assert!(Filter::parse(&json!("just a string")).is_err());
        assert!(Filter::parse(&json!({"$and": "not-an-array"})).is_err());
        assert!(Filter::parse(&json!({"a": {"exists": "yes"}})).is_err());
        assert!(Filter::parse(&json!({"a": {"unknown_op": 1}})).is_err());
        assert!(Filter::parse(&json!({"year": {"gte": 2024, "typo": 1}})).is_err());
    }

    #[test]
    fn empty_filter_is_true() {
        assert!(Filter::parse(&json!(null)).unwrap().is_true());
        assert!(Filter::parse(&json!({})).unwrap().is_true());
    }
}
