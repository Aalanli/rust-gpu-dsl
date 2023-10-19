use std::{hash::Hash, fmt::Display, marker::PhantomData};
mod map_list;
mod node_list;

pub use map_list::{MapDoubleList, MapListKey, VecDoubleList, VecListKey};

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

#[cfg(test)]
pub mod test {
    use super::*;
    fn test_list(lst: &mut impl DoubleList<i32>) {
        
    }
}