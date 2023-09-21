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

    pub fn get_trait<'a, TRAIT: 'static + ?Sized>(&self, a: &'a dyn Any) -> Option<&'a TRAIT> {
        use std::any::TypeId;
        let f = self
            .trait_converters
            .get(&(a.type_id(), TypeId::of::<TRAIT>()))?
            .downcast_ref::<fn(&dyn Any) -> &TRAIT>()?;
        Some(f(a))
    }

    pub fn register_trait<T: 'static, TRAIT: 'static + ?Sized>(&mut self, f: fn(&dyn Any) -> &TRAIT) {
        use std::any::TypeId;

        self.trait_converters
            .insert((TypeId::of::<T>(), TypeId::of::<TRAIT>()), Box::new(f));
    }
}

pub struct TraitRegistry3 {
    converters: Vec<(*const (), TypeId)>,
    cache: RefCell<HashMap<(TypeId, TypeId), *const ()>>
}

impl TraitRegistry3 {
    pub fn new() -> Self {
        TraitRegistry3 {
            converters: vec![],
            cache: RefCell::new(HashMap::new())
        }
    }

    pub fn get_trait<'a, TRAIT: 'static + ?Sized>(&self, a: &'a dyn Any) -> Option<&'a TRAIT> {
        if let Some(f) = self.cache.borrow().get(&(a.type_id(), TypeId::of::<TRAIT>())) {
            unsafe {
                let fp: fn(&dyn Any) -> Option<&TRAIT> = std::mem::transmute(*f);
                // println!("f1 {:?}", fp);
                return (fp)(a);
            }
        }
        for (f, id) in self.converters.iter() {
            if id == &TypeId::of::<TRAIT>() {
                unsafe {
                    let fp: fn(&dyn Any) -> Option<&TRAIT> = std::mem::transmute(*f);
                    // let fp = *f1;
                    // println!("f {:?}", f1);
                    // println!("f {:?}", fp);
                    let val = fp(a);
                    // println!("fin");
                    if val.is_some() {
                        self.cache.borrow_mut().insert((a.type_id(), TypeId::of::<TRAIT>()), std::mem::transmute(fp));
                        return val;
                    }
                }
            }
        }
        None
    }

    pub fn register_trait<TRAIT: 'static + ?Sized>(&mut self, f: fn(&dyn Any) -> Option<&TRAIT>) {
        unsafe {
            // println!("s {:?}", f);
            // println!("s {:?}", &f);
            // println!("s {:?}\n", std::mem::transmute::<_, *const ()>(f));
            
            self.converters.push((std::mem::transmute(f), TypeId::of::<TRAIT>()))
        }
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

#[cfg(test)]
mod test {
    use super::*;
    trait Trait1 {
        fn f1(&self, a: i32) -> i32;
        fn f2(&self) -> &str;
    }
    
    trait Trait2 {
        fn f1(&self) -> i32;
        fn f2(&self) -> &str;
        fn f3(&self) -> usize;
    }
    
    
    trait Trait3 {
        fn f1(&self) -> i32;
        fn f2(&self) -> &str;
        fn f3(&self) -> &usize;
        fn f4(&self) -> [u32; 2];
        fn f5(&self, a: usize) -> usize;
        fn f6<'a>(&self, b: &'a str) -> &'a [u8];
    }
    
    
    macro_rules! make_type {
        ($type:ident $( ,$t_def:tt)*) => {
            struct $type (usize $( ,$t_def)*);
    
            impl Trait1 for $type {
                fn f1(&self, a: i32) -> i32 {
                    (1 + a).div_euclid(2)
                }
                fn f2(&self) -> &str {
                    stringify!("type {}, trait2, fn2", $type)
                }
            }
    
            impl Trait2 for $type {
                fn f1(&self) -> i32 {
                    self.0 as i32
                }
                fn f2(&self) -> &str {
                    stringify!("type {}, trait2, fn2", $type)
                }
                fn f3(&self) -> usize {
                    self.0
                }
            }
    
            impl Trait3 for $type {
                fn f1(&self) -> i32 {
                    self.0 as i32
                }
                fn f2(&self) -> &str {
                    stringify!("type {}, trait3, fn2", $type)
                }
                fn f3(&self) -> &usize {
                    &self.0
                }
                fn f4(&self) -> [u32; 2] {
                    [1, self.0 as u32]
                }
                fn f5(&self, a: usize) -> usize {
                    self.0 + a
                }
                fn f6<'a>(&self, b: &'a str) -> &'a [u8] {
                    b.as_bytes()
                }
            }
            
        };
    }
    
    make_type!(A);
    make_type!(B, u32);
    make_type!(C, u32, f32);
    make_type!(D, u64, usize);
    make_type!(E, u64, usize, u32);

    #[test]
    fn test_trait_registry() {
        let mut reg = TraitRegistry::new();
        reg.register_trait::<A, _>(|a: &dyn Any| a.downcast_ref::<A>().unwrap() as &dyn Trait1);
        reg.register_trait::<A, _>(|a: &dyn Any| a.downcast_ref::<A>().unwrap() as &dyn Trait2);
        reg.register_trait::<A, _>(|a: &dyn Any| a.downcast_ref::<A>().unwrap() as &dyn Trait3);
        reg.register_trait::<B, _>(|a: &dyn Any| a.downcast_ref::<B>().unwrap() as &dyn Trait1);
        reg.register_trait::<B, _>(|a: &dyn Any| a.downcast_ref::<B>().unwrap() as &dyn Trait2);
        reg.register_trait::<B, _>(|a: &dyn Any| a.downcast_ref::<B>().unwrap() as &dyn Trait3);
        reg.register_trait::<C, _>(|a: &dyn Any| a.downcast_ref::<C>().unwrap() as &dyn Trait1);
        reg.register_trait::<C, _>(|a: &dyn Any| a.downcast_ref::<C>().unwrap() as &dyn Trait2);
        reg.register_trait::<C, _>(|a: &dyn Any| a.downcast_ref::<C>().unwrap() as &dyn Trait3);
        reg.register_trait::<D, _>(|a: &dyn Any| a.downcast_ref::<D>().unwrap() as &dyn Trait1);
        reg.register_trait::<D, _>(|a: &dyn Any| a.downcast_ref::<D>().unwrap() as &dyn Trait2);
        reg.register_trait::<D, _>(|a: &dyn Any| a.downcast_ref::<D>().unwrap() as &dyn Trait3);
        reg.register_trait::<E, _>(|a: &dyn Any| a.downcast_ref::<E>().unwrap() as &dyn Trait1);
        reg.register_trait::<E, _>(|a: &dyn Any| a.downcast_ref::<E>().unwrap() as &dyn Trait2);
        reg.register_trait::<E, _>(|a: &dyn Any| a.downcast_ref::<E>().unwrap() as &dyn Trait3);
        let work_list = vec![
            Box::new(A(1)) as Box<dyn Any>,
            Box::new(A(2)) as Box<dyn Any>,
            Box::new(A(3)) as Box<dyn Any>,
            Box::new(B(4, 3)) as Box<dyn Any>,
            Box::new(B(5, 3)) as Box<dyn Any>,
            Box::new(B(6, 3)) as Box<dyn Any>,
            Box::new(C(7, 4, 5.0)) as Box<dyn Any>,
            Box::new(C(8, 4, 5.0)) as Box<dyn Any>,
            Box::new(C(9, 4, 5.0)) as Box<dyn Any>,
            Box::new(D(10, 1, 25)) as Box<dyn Any>,
            Box::new(D(11, 1, 25)) as Box<dyn Any>,
            Box::new(D(12, 1, 25)) as Box<dyn Any>,
            Box::new(E(13, 63, 123, 63)) as Box<dyn Any>,
            Box::new(E(14, 63, 123, 63)) as Box<dyn Any>,
            Box::new(E(15, 63, 123, 63)) as Box<dyn Any>,
        ];


        let mut res = 1;
        for a in work_list.iter() {
            let inner: &dyn Any = &**a;
            let f = reg.get_trait::<dyn Trait1>(inner).unwrap();
            res += f.f1(1);
            let f = reg.get_trait::<dyn Trait2>(inner).unwrap();
            res += f.f1();
            let f = reg.get_trait::<dyn Trait3>(inner).unwrap();
            res += f.f1();
        }
    }

    #[test]
    fn test_dyn_cast() {
        let a = A(1);
        let ad: &dyn Any = &a;
        assert!(TypeId::of::<A>() == ad.type_id());
        let ad: Box<dyn Any> = Box::new(a);
        let inner: &dyn Any = &*ad;
        let at: &A = inner.downcast_ref::<A>().unwrap();
        assert!(TypeId::of::<A>() == inner.type_id());        
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

    #[test]
    fn test_trait_registry3() {
        struct A;
        trait TA {}
        impl TA for A {}
        let mut reg = TraitRegistry3::new();
        reg.register_trait(|a: &dyn Any| a.downcast_ref::<A>().map(|x| x as &dyn TA));
        let a = A;
        let ta = reg.get_trait::<dyn TA>(&a).unwrap();
        let ta = reg.get_trait::<dyn TA>(&a).unwrap();
    }

    #[test]
    fn test_trait_registry3v2() {
        let mut reg = TraitRegistry3::new();
        reg.register_trait(|a: &dyn Any| a.downcast_ref::<A>().map(|x| x as &dyn Trait1));
        reg.register_trait(|a: &dyn Any| a.downcast_ref::<A>().map(|x| x as &dyn Trait2));
        reg.register_trait(|a: &dyn Any| a.downcast_ref::<A>().map(|x| x as &dyn Trait3));
        reg.register_trait(|a: &dyn Any| a.downcast_ref::<B>().map(|x| x as &dyn Trait1));
        reg.register_trait(|a: &dyn Any| a.downcast_ref::<B>().map(|x| x as &dyn Trait2));
        reg.register_trait(|a: &dyn Any| a.downcast_ref::<B>().map(|x| x as &dyn Trait3));
        reg.register_trait(|a: &dyn Any| a.downcast_ref::<C>().map(|x| x as &dyn Trait1));
        reg.register_trait(|a: &dyn Any| a.downcast_ref::<C>().map(|x| x as &dyn Trait2));
        reg.register_trait(|a: &dyn Any| a.downcast_ref::<C>().map(|x| x as &dyn Trait3));
        reg.register_trait(|a: &dyn Any| a.downcast_ref::<D>().map(|x| x as &dyn Trait1));
        reg.register_trait(|a: &dyn Any| a.downcast_ref::<D>().map(|x| x as &dyn Trait2));
        reg.register_trait(|a: &dyn Any| a.downcast_ref::<D>().map(|x| x as &dyn Trait3));
        reg.register_trait(|a: &dyn Any| a.downcast_ref::<E>().map(|x| x as &dyn Trait1));
        reg.register_trait(|a: &dyn Any| a.downcast_ref::<E>().map(|x| x as &dyn Trait2));
        reg.register_trait(|a: &dyn Any| a.downcast_ref::<E>().map(|x| x as &dyn Trait3));

        let work_list = vec![
            Box::new(A(1)) as Box<dyn Any>,
            Box::new(A(2)) as Box<dyn Any>,
            Box::new(A(3)) as Box<dyn Any>,
            Box::new(B(4, 3)) as Box<dyn Any>,
            Box::new(B(5, 3)) as Box<dyn Any>,
            Box::new(B(6, 3)) as Box<dyn Any>,
            Box::new(C(7, 4, 5.0)) as Box<dyn Any>,
            Box::new(C(8, 4, 5.0)) as Box<dyn Any>,
            Box::new(C(9, 4, 5.0)) as Box<dyn Any>,
            Box::new(D(10, 1, 25)) as Box<dyn Any>,
            Box::new(D(11, 1, 25)) as Box<dyn Any>,
            Box::new(D(12, 1, 25)) as Box<dyn Any>,
            Box::new(E(13, 63, 123, 63)) as Box<dyn Any>,
            Box::new(E(14, 63, 123, 63)) as Box<dyn Any>,
            Box::new(E(15, 63, 123, 63)) as Box<dyn Any>,
        ];

        let mut i = 0;
        let mut res = 1;
        for a in work_list.iter() {
            println!("{i}");
            i += 1;
            let inner: &dyn Any = &**a;
            let f = reg.get_trait::<dyn Trait1>(inner).unwrap();
            res += f.f1(1);
            let f = reg.get_trait::<dyn Trait2>(inner).unwrap();
            res += f.f1();
            let f = reg.get_trait::<dyn Trait3>(inner).unwrap();
            res += f.f1();
        }
        
    }

    #[test]
    fn test_fn_ptr() {
        let mut hashmap = HashMap::new();
        let f: fn(i32) -> i32 = |i| i * 2;
        unsafe {
            let f_ptr: *const () = std::mem::transmute(&f);
            hashmap.insert("hi", &f_ptr);
            let f1: *const fn(i32) -> i32 = std::mem::transmute(f_ptr);
            
            println!("hi");
            let f = *f1;
            println!("hi");
            println!("{:?}, {:?}", f, f1);
            println!("{}, {}", f(1), (*f1)(1));
            let f2: *const fn(i32) -> i32 = std::mem::transmute(**(hashmap.get("hi").unwrap()));
            println!("{}", (*f2)(1));
        }
    }
}


