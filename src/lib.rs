use std::any::Any;
use std::cell::{Ref, RefCell};
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::thread::Scope;

use anyhow::{Error, Result};

pub mod lang;
pub mod utils;
mod linear_ir;

struct TraitRegistry {
    trait_converters: HashMap<(std::any::TypeId, std::any::TypeId), Box<dyn Any>>,
}

impl TraitRegistry {
    fn get_trait<'a, T: 'static, F: 'static + ?Sized>(&self, a: &'a T) -> Option<&'a F> {
        use std::any::TypeId;
        let f = self
            .trait_converters
            .get(&(TypeId::of::<T>(), TypeId::of::<F>()))?
            .downcast_ref::<fn(&T) -> &F>()?;
        println!("{:?}", f.type_id());
        Some(f(a))
    }

    fn register_trait<T: 'static, F: 'static + ?Sized>(&mut self, f: fn(&T) -> &F) {
        use std::any::TypeId;
        println!("{:?}", TypeId::of::<fn(&T) -> &F>());

        self.trait_converters
            .insert((TypeId::of::<T>(), TypeId::of::<F>()), Box::new(f));
    }
}

pub trait AsAny {
    fn as_any(&self) -> &dyn Any;
}

impl<T: 'static + Sized> AsAny for T {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

pub trait Operation: AsAny {
    fn uses(&self) -> i32;
}

struct OpTraitRegistry {
    converters: HashMap<(std::any::TypeId, std::any::TypeId), Box<dyn Any>>,
}

impl OpTraitRegistry {
    fn new() -> Self {
        OpTraitRegistry {
            converters: HashMap::new(),
        }
    }

    fn register_trait<OP: Operation + 'static, TRAIT: 'static + ?Sized>(
        &mut self,
        f: fn(&dyn Operation) -> Option<&TRAIT>,
    ) -> Result<()> {
        let id = std::any::TypeId::of::<OP>();
        let trait_id = std::any::TypeId::of::<TRAIT>();

        if self.converters.contains_key(&(id, trait_id)) {
            return Err(Error::msg("Converter already registered"));
        }
        self.converters.insert((id, trait_id), Box::new(f));
        Ok(())
    }

    fn get_trait<'a, TRAIT: 'static + ?Sized>(&self, op: &'a dyn Operation) -> Option<&'a TRAIT> {
        let id = op.as_any().type_id();
        let trait_id = std::any::TypeId::of::<TRAIT>();
        println!("{:?} {:?}", id, trait_id);
        let f = self
            .converters
            .get(&(id, trait_id))?
            .downcast_ref::<fn(&dyn Operation) -> Option<&TRAIT>>()?;
        f(op)
    }
}
