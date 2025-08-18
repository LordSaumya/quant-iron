use std::sync::{Arc, Mutex};

/// A struct representing a set of mutable parameters for a parametrised gate.
/// Cloning this struct is cheap and allows multiple gates to share the same underlying parameters.
///
/// # Fields
/// - `values`: An `Arc<Mutex<[f64; N]>>` that holds the parameter values.
///
/// # Generic Parameters
/// - `N`: The number of values in the parameter array.
#[derive(Clone)]
pub struct Parameter<const N: usize> {
    /// The values of the parameters
    values: Arc<Mutex<[f64; N]>>,
}

impl<const N: usize> Parameter<N> {
    /// Creates a new `Parameter` instance with the given initial values.
    ///
    /// # Arguments
    ///
    /// * `initial_values` - An array of initial parameter values.
    ///
    /// # Returns
    ///
    /// A new `Parameter` instance.
    pub fn new(initial_values: [f64; N]) -> Self {
        Self {
            values: Arc::new(Mutex::new(initial_values)),
        }
    }

    /// Creates a new `Parameter` instance as an independent copy of the given parameter.
    ///
    /// # Returns
    ///
    /// A new `Parameter` instance.
    pub fn deep_clone(&self) -> Self {
        Self {
            values: Arc::new(Mutex::new(self.get())),
        }
    }

    /// Gets the current values of the parameters.
    /// This will lock the underlying mutex.
    ///
    /// # Returns
    ///
    /// The current parameter values.
    ///
    /// If the mutex is poisoned, the function will return the current values.
    pub fn get(&self) -> [f64; N] {
        match self.values.lock() {
            Ok(guard) => *guard,
            Err(poisoned) => *poisoned.into_inner(),
        }
    }

    /// Sets the values of the parameters.
    /// This will lock the underlying mutex.
    pub fn set(&self, new_values: [f64; N]) {
        let mut guard = match self.values.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        *guard = new_values;
    }
}

impl<const N: usize> std::fmt::Debug for Parameter<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let values = self.get();
        write!(f, "Parameter({:?})", values)
    }
}