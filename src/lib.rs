use std::collections::HashMap;
use std::any::Any;

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

