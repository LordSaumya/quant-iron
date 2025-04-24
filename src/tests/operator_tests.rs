use num_complex::Complex;
use crate::{errors::Error, components::{state::{State, ChainableState}, operator::Operator}};

#[test]
fn test_state_hadamard_success() {
    // h(|0>) = |+>
    // h(|1>) = |-> 
    // h(|+>) = |0>
    // h(|->) = |1>
    let zero_state: State = State::new_zero(1).unwrap();
    let one_state: State = State::new_basis_n(1, 1).unwrap();
    let plus_state: State = State::new_plus(1).unwrap();
    let minus_state: State = State::new_minus(1).unwrap();

    assert_eq!(zero_state.h(0).unwrap(), plus_state);
    assert_eq!(one_state.h(0).unwrap(), minus_state);
    assert_eq!(plus_state.h(0).unwrap(), zero_state);
    assert_eq!(minus_state.h(0).unwrap(), one_state);
}