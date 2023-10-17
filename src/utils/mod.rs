use std::hash::Hash;
mod map_list;
mod vec_list;
mod node_list;

pub trait DoubleList<T> {
    type Key: Clone + Eq + Hash;
    fn is_valid_key(&self, key: &Self::Key) -> bool;
    fn front(&self) -> Option<Self::Key>;
    fn back(&self) -> Option<Self::Key>;
    fn push_front(&mut self, item: T) -> Self::Key;
    fn push_back(&mut self, item: T) -> Self::Key;

    fn prev(&self, key: &Self::Key) -> Option<&Self::Key>;
    fn next(&self, key: &Self::Key) -> Option<&Self::Key>;

    fn get(&self, key: &Self::Key) -> Option<&T>;
    fn get_mut(&mut self, key: &Self::Key) -> Option<&mut T>;

    fn remove(&mut self, key: &Self::Key) -> Option<T>;
    fn insert_before(&mut self, key: &Self::Key, item: T) -> Option<Self::Key>;
    fn insert_after(&mut self, key: &Self::Key, item: T) -> Option<Self::Key>;
}
