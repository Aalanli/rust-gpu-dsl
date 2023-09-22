use criterion::*;
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

fn make_dyn_objs(i: u64) -> Vec<Box<dyn Any>> {
    let mut work_list = vec![];
    for i in 0..black_box(i) {
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

fn criterion_benchmark(c: &mut Criterion) {
    use rust_gpu_dsl::{TraitRegistry, TraitRegistry2, TraitRegistry3};

    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
    let mut group = c.benchmark_group("trait_reg");
    group.plot_config(plot_config);
    let reg = {
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
        reg 
    };

    let reg2 = {
        let mut reg2 = TraitRegistry2::new();
        reg2.register_trait(|a: &A| a as &dyn Trait1);
        reg2.register_trait(|a: &A| a as &dyn Trait2);
        reg2.register_trait(|a: &A| a as &dyn Trait3);
        reg2.register_trait(|a: &B| a as &dyn Trait1);
        reg2.register_trait(|a: &B| a as &dyn Trait2);
        reg2.register_trait(|a: &B| a as &dyn Trait3);
        reg2.register_trait(|a: &C| a as &dyn Trait1);
        reg2.register_trait(|a: &C| a as &dyn Trait2);
        reg2.register_trait(|a: &C| a as &dyn Trait3);
        reg2.register_trait(|a: &D| a as &dyn Trait1);
        reg2.register_trait(|a: &D| a as &dyn Trait2);
        reg2.register_trait(|a: &D| a as &dyn Trait3);
        reg2.register_trait(|a: &E| a as &dyn Trait1);
        reg2.register_trait(|a: &E| a as &dyn Trait2);
        reg2.register_trait(|a: &E| a as &dyn Trait3);
        reg2
    };

    let reg3 = {
        let mut reg3 = TraitRegistry3::new();
        reg3.register_trait(|a: &dyn Any| a.downcast_ref::<A>().map(|x| x as &dyn Trait1));
        reg3.register_trait(|a: &dyn Any| a.downcast_ref::<A>().map(|x| x as &dyn Trait2));
        reg3.register_trait(|a: &dyn Any| a.downcast_ref::<A>().map(|x| x as &dyn Trait3));
        reg3.register_trait(|a: &dyn Any| a.downcast_ref::<B>().map(|x| x as &dyn Trait1));
        reg3.register_trait(|a: &dyn Any| a.downcast_ref::<B>().map(|x| x as &dyn Trait2));
        reg3.register_trait(|a: &dyn Any| a.downcast_ref::<B>().map(|x| x as &dyn Trait3));
        reg3.register_trait(|a: &dyn Any| a.downcast_ref::<C>().map(|x| x as &dyn Trait1));
        reg3.register_trait(|a: &dyn Any| a.downcast_ref::<C>().map(|x| x as &dyn Trait2));
        reg3.register_trait(|a: &dyn Any| a.downcast_ref::<C>().map(|x| x as &dyn Trait3));
        reg3.register_trait(|a: &dyn Any| a.downcast_ref::<D>().map(|x| x as &dyn Trait1));
        reg3.register_trait(|a: &dyn Any| a.downcast_ref::<D>().map(|x| x as &dyn Trait2));
        reg3.register_trait(|a: &dyn Any| a.downcast_ref::<D>().map(|x| x as &dyn Trait3));
        reg3.register_trait(|a: &dyn Any| a.downcast_ref::<E>().map(|x| x as &dyn Trait1));
        reg3.register_trait(|a: &dyn Any| a.downcast_ref::<E>().map(|x| x as &dyn Trait2));
        reg3.register_trait(|a: &dyn Any| a.downcast_ref::<E>().map(|x| x as &dyn Trait3));
        reg3
    };
    
    for i in [1u64, 10, 100, 1000, 10000, 100000, 1000000].into_iter() {
        group.bench_function(BenchmarkId::from_parameter(i), |b| {
            let mut work_list = make_dyn_objs(i);
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
    }
    group.finish();

    let work_list = make_dyn_objs(100);

    c.bench_function("registry2", |b| {
        b.iter(|| {
            let mut res = 1;
            for a in work_list.iter() {
                let inner: &dyn Any = &**a;

                let f = reg2.get_trait_any::<dyn Trait1>(inner).unwrap();
                res += f.f1(1);
                let f = reg2.get_trait_any::<dyn Trait2>(inner).unwrap();
                res += f.f1();
                let f = reg2.get_trait_any::<dyn Trait3>(inner).unwrap();
                res += f.f1();
                
                
            }
            res
        })
    });


    
    c.bench_function("registry3", |b| {
        b.iter(|| {
            let mut res = 1;
            for a in work_list.iter() {
                let inner: &dyn Any = &**a;
                let f = reg3.get_trait::<dyn Trait1>(inner).unwrap();
                res += f.f1(1);
                let f = reg3.get_trait::<dyn Trait2>(inner).unwrap();
                res += f.f1();
                let f = reg3.get_trait::<dyn Trait3>(inner).unwrap();
                res += f.f1();
            }
            res
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
