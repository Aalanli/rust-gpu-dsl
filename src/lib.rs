use std::any::{Any, TypeId};

use std::collections::{HashMap};






pub mod lang;
pub mod utils;
mod linear_ir;

pub struct TraitRegistry {
    trait_converters: HashMap<(std::any::TypeId, std::any::TypeId), Box<dyn for <'a> Fn(&'a dyn Any) -> *const ()>>,
}

impl TraitRegistry {
    pub fn new() -> Self {
        TraitRegistry
     {
            trait_converters: HashMap::new(),
        }
    }

    pub fn get_trait<'a, TRAIT: 'static + ?Sized>(&self, a: &'a dyn Any) -> Option<&'a TRAIT> {
        let f = self
            .trait_converters
            .get(&(a.type_id(), TypeId::of::<TRAIT>()))?;
        unsafe {
            let a = f(a);
            let b = Box::<&TRAIT>::from_raw(a as *mut &TRAIT);
            Some(*b)
        }
    }

    pub fn register_trait<F: 'static, T: 'static, TRAIT: 'static + ?Sized>(&mut self, f: F)
    where F: for<'a> Fn(&'a T) -> &'a TRAIT
    {
        let f1 = move |a: &dyn Any| {
            let a = a.downcast_ref::<T>().unwrap();
            let any = f(a);
            let ptr = Box::new(any);
            let ptr_of = Box::<&TRAIT>::into_raw(ptr);
            ptr_of as *const ()
        };
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

    fn make_dyn_objs(i: u64) -> Vec<Box<dyn Any>> {
        let mut work_list = vec![];
        for i in 0..i {
            let i = i as usize;
            let j = i % 5;
            match j {
                0 => work_list.push(Box::new(A(i)) as Box<dyn Any>),
                1 => work_list.push(Box::new(B(i, i as u32)) as Box<dyn Any>),
                2 => work_list.push(Box::new(C(i, i as u32, i as f32)) as Box<dyn Any>),
                3 => work_list.push(Box::new(D(i, i as u64, i)) as Box<dyn Any>),
                _ => work_list.push(Box::new(E(i, i as u64, i, i as u32)) as Box<dyn Any>),
            }
        }
        work_list
    }

    #[test]
    fn test_trait_registry() {
        struct A1;
        trait TA {}
        impl TA for A1 {}
        let mut reg = TraitRegistry::new();
        reg.register_trait(|a: &A1| a as &dyn TA);
        let a = A1;
        let a: &dyn Any = &a;
        let _ta = reg.get_trait::<dyn TA>(a).unwrap();
        let work_list = make_dyn_objs(100);
        reg.register_trait(|a: &A| a as &dyn Trait1);
        reg.register_trait(|a: &B| a as &dyn Trait1);
        reg.register_trait(|a: &C| a as &dyn Trait1);
        reg.register_trait(|a: &D| a as &dyn Trait1);
        reg.register_trait(|a: &E| a as &dyn Trait1);
        reg.register_trait(|a: &A| a as &dyn Trait2);
        reg.register_trait(|a: &B| a as &dyn Trait2);
        reg.register_trait(|a: &C| a as &dyn Trait2);
        reg.register_trait(|a: &D| a as &dyn Trait2);
        reg.register_trait(|a: &E| a as &dyn Trait2);
        reg.register_trait(|a: &A| a as &dyn Trait3);
        reg.register_trait(|a: &B| a as &dyn Trait3);
        reg.register_trait(|a: &C| a as &dyn Trait3);
        reg.register_trait(|a: &D| a as &dyn Trait3);
        reg.register_trait(|a: &E| a as &dyn Trait3);
        for a in work_list {
            let _ta = reg.get_trait::<dyn Trait1>(&*a).unwrap();
            let _ta = reg.get_trait::<dyn Trait2>(&*a).unwrap();
            let _ta = reg.get_trait::<dyn Trait3>(&*a).unwrap();
        }
        
    }
}


