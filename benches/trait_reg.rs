use criterion::{black_box, criterion_group, criterion_main, Criterion};
use std::any::Any;

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


fn criterion_benchmark(c: &mut Criterion) {
    use rust_gpu_dsl::{TraitRegistry, TraitRegistry2};
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
    let work_list = black_box(vec![
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
    ]);

    c.bench_function("registry1", |b| {
        b.iter(|| {
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
            res
        })
    });


    let mut reg = TraitRegistry2::new();
    reg.register_trait(|a: &A| a as &dyn Trait1);
    reg.register_trait(|a: &A| a as &dyn Trait2);
    reg.register_trait(|a: &A| a as &dyn Trait3);
    reg.register_trait(|a: &B| a as &dyn Trait1);
    reg.register_trait(|a: &B| a as &dyn Trait2);
    reg.register_trait(|a: &B| a as &dyn Trait3);
    reg.register_trait(|a: &C| a as &dyn Trait1);
    reg.register_trait(|a: &C| a as &dyn Trait2);
    reg.register_trait(|a: &C| a as &dyn Trait3);
    reg.register_trait(|a: &D| a as &dyn Trait1);
    reg.register_trait(|a: &D| a as &dyn Trait2);
    reg.register_trait(|a: &D| a as &dyn Trait3);
    reg.register_trait(|a: &E| a as &dyn Trait1);
    reg.register_trait(|a: &E| a as &dyn Trait2);
    reg.register_trait(|a: &E| a as &dyn Trait3);

    c.bench_function("registry2", |b| {
        b.iter(|| {
            let mut res = 1;
            for a in work_list.iter() {
                let inner: &dyn Any = &**a;

                let f = reg.get_trait_any::<dyn Trait1>(inner).unwrap();
                res += f.f1(1);
                let f = reg.get_trait_any::<dyn Trait2>(inner).unwrap();
                res += f.f1();
                let f = reg.get_trait_any::<dyn Trait3>(inner).unwrap();
                res += f.f1();
                
                
            }
            res
        })
    });
    
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
