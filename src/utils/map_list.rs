use std::collections::HashMap;
use super::DoubleList;

#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash)]
pub struct MapListKey {
    generation: usize,
}

#[derive(Clone, Debug)]
struct ListItem<T> {
    item: T,
    prev: usize, // prev does not exist if prev == front
    next: usize, // next does not exist if next == back
}

#[derive(Clone, Debug)]
pub struct MapDoubleList<T> {
    generation: usize,
    storage: HashMap<usize, ListItem<T>>,
    front: usize,
    back: usize,
}


impl<T> MapDoubleList<T> {
    pub fn new() -> Self {
        Self {
            generation: 0,
            storage: HashMap::new(),
            front: 0,
            back: 0,
        }
    }

    pub fn clear(&mut self) {
        self.generation = 0;
        self.storage.clear();
        self.front = 0;
        self.back = 0;
    }

    pub fn prev(&self, key: &MapListKey) -> Option<MapListKey> {
        if let Some(x) = self.storage.get(&key.generation) {
            if x.prev == self.front {
                None
            } else {
                Some(MapListKey { generation: x.prev })
            }
        } else {
            None
        }
    }
    pub fn next(&self, key: &MapListKey) -> Option<MapListKey> {
        if let Some(x) = self.storage.get(&key.generation) {
            if x.next == self.back {
                None
            } else {
                Some(MapListKey { generation: x.next })
            }
        } else {
            None
        }
    }
    pub fn front(&self) -> Option<MapListKey> {
        if self.storage.len() > 0 {
            Some(MapListKey { generation: self.front })
        } else {
            None
        }
    }

    pub fn back(&self) -> Option<MapListKey> {
        if self.storage.len() > 0 {
            Some(MapListKey { generation: self.back })
        } else {
            None
        }
    }

    pub fn push_front(&mut self, item: T) -> MapListKey { 
        let new_key = MapListKey { generation: self.generation };
        self.generation += 1;
        let mut item = ListItem {
            item,
            prev: new_key.generation,
            next: new_key.generation,
        };

        if let Some(front) = self.storage.get_mut(&self.front) {
            front.prev = new_key.generation;
            item.next = self.front;
            self.front = new_key.generation;
        } else {
            self.front = new_key.generation;
            self.back = new_key.generation;
        }
        self.storage.insert(new_key.generation, item);
        new_key
    }

    pub fn push_back(&mut self, item: T) -> MapListKey { 
        let new_key = MapListKey { generation: self.generation };
        self.generation += 1;
        let mut item = ListItem {
            item,
            prev: new_key.generation,
            next: new_key.generation,
        };

        if let Some(back) = self.storage.get_mut(&self.back) {
            back.next = new_key.generation;
            item.prev = self.back;
            self.back = new_key.generation;
        } else {
            self.front = new_key.generation;
            self.back = new_key.generation;
        }
        self.storage.insert(new_key.generation, item);
        new_key
    }

    pub fn is_valid_key(&self, key: &MapListKey) -> bool {
        self.storage.contains_key(&key.generation)
    }

    pub fn get(&self, key: &MapListKey) -> Option<&T> {
        self.storage.get(&key.generation).map(|x| &x.item)
    }

    pub fn get_mut(&mut self, key: &MapListKey) -> Option<&mut T> {
        self.storage.get_mut(&key.generation).map(|x| &mut x.item)
    }

    fn set_prev(&mut self, key: &MapListKey, prev: usize) {
        if let Some(x) = self.storage.get_mut(&key.generation) {
            x.prev = prev;
        }
    }

    fn set_next(&mut self, key: &MapListKey, next: usize) {
        if let Some(x) = self.storage.get_mut(&key.generation) {
            x.next = next;
        }
    }

    pub fn remove(&mut self, key: &MapListKey) -> Option<T> {
        if let Some(x) = self.storage.remove(&key.generation) {
            self.set_next(&MapListKey { generation: x.prev }, x.next);
            self.set_prev(&MapListKey { generation: x.next }, x.prev);
            if self.front == key.generation {
                self.front = x.next;
            }
            if self.back == key.generation {
                self.back = x.prev;
            }
            return Some(x.item);
        }
        None
    }

    pub fn insert_before(&mut self, key: &MapListKey, item: T) -> Option<T> {
        if let Some(cur) = self.storage.get_mut(&key.generation) {
            let new_key = MapListKey { generation: self.generation };
            self.generation += 1;
            let old_prev = cur.prev;
            cur.prev = new_key.generation;
            
            let prev_gen = if key.generation == self.front {
                self.front = new_key.generation;
                new_key.generation
            } else {
                let Some(prev) = self.storage.get_mut(&old_prev) else { unreachable!() };
                prev.next = new_key.generation;
                old_prev
            };
            self.storage.insert(new_key.generation, ListItem {
                item,
                prev: prev_gen,
                next: key.generation,
            });
        }
        None
    }

    pub fn insert_after(&mut self, key: &MapListKey, item: T) -> Option<T> {
        if let Some(cur) = self.storage.get_mut(&key.generation) {
            let new_key = MapListKey { generation: self.generation };
            self.generation += 1;
            let old_next = cur.next;
            cur.prev = new_key.generation;
            
            let next_gen = if key.generation == self.back {
                self.back = new_key.generation;
                new_key.generation
            } else {
                let Some(next) = self.storage.get_mut(&old_next) else { unreachable!() };
                next.prev = new_key.generation;
                old_next
            };
            self.storage.insert(new_key.generation, ListItem {
                item,
                prev: key.generation,
                next: next_gen,
            });
        }
        None
    }
}

impl<T> DoubleList<T> for MapDoubleList<T> {
    type Key = MapListKey;

    fn is_valid_key(&self, key: &Self::Key) -> bool {
        Self::is_valid_key(self, key)
    }

    fn front(&self) -> Option<Self::Key> {
        Self::front(self)
    }

    fn back(&self) -> Option<Self::Key> {
        Self::back(self)
    }

    fn push_front(&mut self, item: T) -> Self::Key {
        Self::push_front(self, item)
    }

    fn push_back(&mut self, item: T) -> Self::Key {
        Self::push_back(self, item)
    }

    fn prev(&self, key: &Self::Key) -> Option<Self::Key> {
        Self::prev(self, key)
    }

    fn next(&self, key: &Self::Key) -> Option<Self::Key> {
        Self::next(self, key)
    }

    fn get(&self, key: &Self::Key) -> Option<&T> {
        Self::get(self, key)
    }

    fn get_mut(&mut self, key: &Self::Key) -> Option<&mut T> {
        Self::get_mut(self, key)
    }

    fn remove(&mut self, key: &Self::Key) -> Option<T> {
        Self::remove(self, key)
    }

    fn insert_before(&mut self, key: &Self::Key, item: T) -> Option<T> {
        Self::insert_before(self, key, item)
    }

    fn insert_after(&mut self, key: &Self::Key, item: T) -> Option<T> {
        Self::insert_after(self, key, item)
    }
}