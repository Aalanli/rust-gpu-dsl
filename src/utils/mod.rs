use std::{hash::Hash, fmt::Display, marker::PhantomData};
mod map_list;
mod node_list;
mod vec_list;

pub use map_list::{MapDoubleList, MapListKey, VecDoubleListV1, VecListKeyV1};
pub use vec_list::{VecDoubleList, VecListKey};

pub trait DoubleList<T> {
    type Key: Clone + Eq + Hash;
    fn is_valid_key(&self, key: &Self::Key) -> bool;
    fn front(&self) -> Option<Self::Key>;
    fn back(&self) -> Option<Self::Key>;
    fn push_front(&mut self, item: T) -> Self::Key;
    fn push_back(&mut self, item: T) -> Self::Key;

    fn prev(&self, key: &Self::Key) -> Option<Self::Key>;
    fn next(&self, key: &Self::Key) -> Option<Self::Key>;

    fn get(&self, key: &Self::Key) -> Option<&T>;
    fn get_mut(&mut self, key: &Self::Key) -> Option<&mut T>;

    fn remove(&mut self, key: &Self::Key) -> Option<T>;
    fn insert_before(&mut self, key: &Self::Key, item: T) -> Option<T>;
    fn insert_after(&mut self, key: &Self::Key, item: T) -> Option<T>;
}

pub struct DoubleListOwnedIter<T, L: DoubleList<T>> {
    list: L,
    _marker: PhantomData<T>
}

impl<T, L: DoubleList<T>> Iterator for DoubleListOwnedIter<T, L> {
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        let k = self.list.front()?;
        self.list.remove(&k)
    }
}

pub struct DoubleListRefIter<'a, T, L: DoubleList<T>> {
    list: &'a L,
    key: Option<L::Key>,
    _marker: PhantomData<&'a T>
}


impl<'a, T, L: DoubleList<T>> Iterator for DoubleListRefIter<'a, T, L> {
    type Item = &'a T;
    fn next(&mut self) -> Option<Self::Item> {
        let new_key = self.key.as_ref()?;
        let item = self.list.get(new_key);
        self.key = self.list.next(new_key);
        item
    }
}


pub fn simple_print_list<T: Display>(list: &impl DoubleList<T>) {
    let mut key = list.front();
    while let Some(k) = key {
        let item = list.get(&k).unwrap();
        print!("{} ", item);
        key = list.next(&k);
    }
    println!();
}

pub struct Printer {

}

impl Printer {
    pub fn new() -> Self { todo!() }
    pub fn line(&mut self, f: impl FnOnce(&mut Self)) { todo!() }
    pub fn indent(&mut self, f: impl FnOnce(&mut Self)) { todo!() }

    pub fn token(&mut self, token: impl Into<String>) { 
        let token: String = token.into();
        assert!(self.is_token(&token), "Token must not contain newlines");
        todo!()
    }
    pub fn tokenlist(&mut self, tokens: impl IntoIterator<Item=impl Into<String>>) { todo!() }

    pub fn is_token(&mut self, token: &str) -> bool { 
        token.chars().all(|c: char| c != '\n')
    }
}

pub trait Attribute: Display + 'static {
    fn as_any(&self) -> &dyn std::any::Any;
}

impl<T: Display + 'static> Attribute for T {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
