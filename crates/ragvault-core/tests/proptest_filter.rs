//! Property tests: the filter parser/evaluator never panics and honors
//! core invariants on arbitrary inputs.

use proptest::prelude::*;
use ragvault_core::Filter;
use serde_json::{json, Value};

/// Arbitrary JSON values (bounded depth/size).
fn arb_json() -> impl Strategy<Value = Value> {
    let leaf = prop_oneof![
        Just(Value::Null),
        any::<bool>().prop_map(Value::from),
        any::<i32>().prop_map(Value::from),
        // Finite doubles only at the leaf level; NaN handling is covered by
        // dedicated unit tests (serde_json cannot represent NaN anyway).
        (-1e12f64..1e12).prop_map(Value::from),
        "[a-z]{0,8}".prop_map(Value::from),
    ];
    leaf.prop_recursive(3, 24, 4, |inner| {
        prop_oneof![
            prop::collection::vec(inner.clone(), 0..4).prop_map(Value::from),
            prop::collection::btree_map("[a-z]{1,6}", inner, 0..4)
                .prop_map(|m| Value::Object(m.into_iter().collect())),
        ]
    })
}

proptest! {
    /// Parsing arbitrary JSON either succeeds or errors — never panics —
    /// and a successful parse never panics during evaluation.
    #[test]
    fn parse_and_eval_never_panic(filter in arb_json(), metadata in arb_json()) {
        if let Ok(parsed) = Filter::parse(&filter) {
            let _ = parsed.matches(&metadata);
        }
    }

    /// De Morgan-ish sanity: not(f) is the complement of f on any metadata.
    #[test]
    fn not_is_complement(field in "[a-z]{1,4}", needle in "[a-z]{0,4}", metadata in arb_json()) {
        let f = Filter::parse(&json!({ field.clone(): needle.clone() })).unwrap();
        let nf = Filter::parse(&json!({ "$not": { field: needle } })).unwrap();
        prop_assert_ne!(f.matches(&metadata), nf.matches(&metadata));
    }

    /// eq and in([x]) agree everywhere.
    #[test]
    fn eq_equals_singleton_in(field in "[a-z]{1,4}", needle in -50i32..50, metadata in arb_json()) {
        let eq = Filter::parse(&json!({ field.clone(): needle })).unwrap();
        let inn = Filter::parse(&json!({ field: { "in": [needle] } })).unwrap();
        prop_assert_eq!(eq.matches(&metadata), inn.matches(&metadata));
    }
}
