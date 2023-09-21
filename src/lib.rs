use std::any::{Any, TypeId};
use std::cell::{Ref, RefCell};
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::thread::Scope;

use anyhow::{Error, Result};

pub mod lang;
pub mod utils;
mod linear_ir;

pub struct TraitRegistry {
    trait_converters: HashMap<(std::any::TypeId, std::any::TypeId), Box<dyn Any>>,
}

impl TraitRegistry {
    pub fn new() -> Self {
        TraitRegistry {
            trait_converters: HashMap::new(),
        }
    }

    pub fn get_trait<'a, TRAIT: 'static + ?Sized, T: 'static>(&self, a: &'a T) -> Option<&'a TRAIT> {
        use std::any::TypeId;
        let f = self
            .trait_converters
            .get(&(TypeId::of::<T>(), TypeId::of::<TRAIT>()))?
            .downcast_ref::<fn(&T) -> &TRAIT>()?;
        Some(f(a))
    }

    pub fn register_trait<T: 'static, F: 'static + ?Sized>(&mut self, f: fn(&T) -> &F) {
        use std::any::TypeId;
        self.trait_converters
            .insert((TypeId::of::<T>(), TypeId::of::<F>()), Box::new(f));
    }
}


pub struct TraitRegistry2 {
    trait_converters: HashMap<(std::any::TypeId, std::any::TypeId), Box<dyn for <'a> Fn(&'a dyn Any) -> *const ()>>,
}

impl TraitRegistry2 {
    pub fn new() -> Self {
        TraitRegistry2 {
            trait_converters: HashMap::new(),
        }
    }

    pub fn get_trait_any<'a, TRAIT: 'static + ?Sized>(&self, a: &'a dyn Any) -> Option<&'a TRAIT> {
        use std::any::TypeId;
        let f = self
            .trait_converters
            .get(&(a.type_id(), TypeId::of::<TRAIT>()))?;
        // let f = f.downcast_ref::<Box<dyn for <'h> Fn(&'h dyn Any) -> &'h TRAIT>>()?;
        // Some(f(a))
        unsafe {
            let a = f(a);
            let b = Box::<&TRAIT>::from_raw(a as *mut &TRAIT);
            Some(*b)
        }
    }

    pub fn get_trait<'a, TRAIT: 'static + ?Sized, T: 'static>(&self, a: &'a T) -> Option<&'a TRAIT> {
        use std::any::TypeId;
        let f = self
            .trait_converters
            .get(&(TypeId::of::<T>(), TypeId::of::<TRAIT>()))?;
        // let f = f.downcast_ref::<Box<dyn for <'h> Fn(&'h dyn Any) -> &'h TRAIT>>()?;
        // Some(f(a))
        unsafe {
            let a = f(a);
            let b = Box::<&TRAIT>::from_raw(a as *mut &TRAIT);
            Some(*b)
        }
    }

    pub fn register_trait<F: 'static, T: 'static, TRAIT: 'static + ?Sized>(&mut self, f: F)
    where F: for<'a> Fn(&'a T) -> &'a TRAIT
    {
        use std::any::TypeId;
        let f1 = move |a: &dyn Any| {
            let a = a.downcast_ref::<T>().unwrap();
            let any = f(a);
            let ptr = Box::new(any);
            let ptr_of = Box::<&TRAIT>::into_raw(ptr);
            ptr_of as *const ()
        };
        // let a: Box<dyn for <'a> Fn(&'a dyn Any) -> *const ()> = Box::new(f1);
        // let b: Box<dyn Any> = Box::new(a);
        let a: Box<dyn for <'a> Fn(&'a dyn Any) -> *const ()> = Box::new(f1);
        self.trait_converters
            .insert((TypeId::of::<T>(), TypeId::of::<TRAIT>()), a);
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

fn test<T: ?Sized>(a: &dyn AsAny, b: &dyn Any, c: &T) {

    unsafe {
        let ta: u128 = std::mem::transmute(a);
        let tb: &dyn AsAny = std::mem::transmute(ta);
    }
}

#[test]
fn test_trait_registry() {
    struct A;
    trait TA {}
    impl TA for A {}
    let mut reg = TraitRegistry::new();
    reg.register_trait(|a: &A| a as &dyn TA);
    let a = A;
    let ta = reg.get_trait::<dyn TA, A>(&a).unwrap();
    println!("{:?}", std::any::TypeId::of::<A>());
}

#[test]
fn test_trait_registry2() {
    struct A;
    trait TA {}
    impl TA for A {}
    let mut reg = TraitRegistry2::new();
    reg.register_trait(|a: &A| a as &dyn TA);
    let a = A;
    let ta = reg.get_trait::<dyn TA, A>(&a).unwrap();
    println!("{:?}", std::any::TypeId::of::<A>());
    let a: &dyn Any = &a;
    println!("{:?}", a.type_id());
    let ta = reg.get_trait_any::<dyn TA>(a).unwrap();
}