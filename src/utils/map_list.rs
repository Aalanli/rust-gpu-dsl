use std::{collections::HashMap, hash::Hash, marker::PhantomData};
use super::DoubleList;


#[derive(Clone, Debug)]
pub struct MapDoubleList<T>(GenericDoubleList<GenericListItem<T>, HashMapStorage<usize, GenericListItem<T>>>);
#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub struct MapListKey(SListKey);
impl<T> MapDoubleList<T> {
    pub fn new() -> Self {
        let hasher = nohash_hasher::BuildNoHashHasher::default();
        MapDoubleList(GenericDoubleList::new(HashMapStorage { hashmap: HashMap::with_hasher(hasher), generation: 0 }))
    }

    pub fn clear(&mut self) {
        self.0.clear();
    }

    pub fn len(&mut self) -> usize {
        self.0.len()
    }

    pub fn is_valid_key(&self, key: &MapListKey) -> bool {
        self.0.is_valid_key(&key.0)
    }

    pub fn front(&self) -> Option<MapListKey> {
        self.0.front().map(|x| MapListKey(x))
    }

    pub fn back(&self) -> Option<MapListKey> {
        self.0.back().map(|x| MapListKey(x))
    }

    pub fn push_front(&mut self, item: T) -> MapListKey {
        MapListKey(self.0.push_front(item))
    }

    pub fn push_back(&mut self, item: T) -> MapListKey {
        MapListKey(self.0.push_back(item))
    }

    pub fn prev(&self, key: &MapListKey) -> Option<MapListKey> {
        self.0.prev(&key.0).map(|x| MapListKey(x))
    }

    pub fn next(&self, key: &MapListKey) -> Option<MapListKey> {
        self.0.next(&key.0).map(|x| MapListKey(x))
    }

    pub fn get(&self, key: &MapListKey) -> Option<&T> {
        self.0.get(&key.0)
    }

    pub fn get_mut(&mut self, key: &MapListKey) -> Option<&mut T> {
        self.0.get_mut(&key.0)
    }

    pub fn remove(&mut self, key: &MapListKey) -> Option<T> {
        self.0.remove(&key.0)
    }

    pub fn insert_before(&mut self, key: &MapListKey, item: T) -> Option<T> {
        self.0.insert_before(&key.0, item)
    }

    pub fn insert_after(&mut self, key: &MapListKey, item: T) -> Option<T> {
        self.0.insert_after(&key.0, item)
    }
}

impl<T> DoubleList<T> for MapDoubleList<T> {
    type Key = MapListKey;
    fn is_valid_key(&self, key: &Self::Key) -> bool {
        self.is_valid_key(key)
    }

    fn front(&self) -> Option<Self::Key> {
        self.front()
    }

    fn back(&self) -> Option<Self::Key> {
        self.back()
    }

    fn push_front(&mut self, item: T) -> Self::Key {
        self.push_front(item)
    }

    fn push_back(&mut self, item: T) -> Self::Key {
        self.push_back(item)
    }

    fn prev(&self, key: &Self::Key) -> Option<Self::Key> {
        self.prev(key)
    }

    fn next(&self, key: &Self::Key) -> Option<Self::Key> {
        self.next(key)
    }

    fn get(&self, key: &Self::Key) -> Option<&T> {
        self.get(key)
    }

    fn get_mut(&mut self, key: &Self::Key) -> Option<&mut T> {
        self.get_mut(key)
    }

    fn remove(&mut self, key: &Self::Key) -> Option<T> {
        self.remove(key)
    }

    fn insert_before(&mut self, key: &Self::Key, item: T) -> Option<T> {
        self.insert_before(key, item)
    }

    fn insert_after(&mut self, key: &Self::Key, item: T) -> Option<T> {
        self.insert_after(key, item)
    }
}

pub struct VecDoubleList<T>(GenericDoubleList<GenericListItem<T>, VecStorage<GenericListItem<T>>>);
#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub struct VecListKey(SListKey);

impl<T> VecDoubleList<T> {
    pub fn new() -> Self {
        VecDoubleList(GenericDoubleList::new(VecStorage::new()))
    }

    pub fn internal_len(&self) -> usize {
        self.0.storage.storage.len()
    }

    pub fn dropped_len(&self) -> usize {
        self.0.storage.freed.len()
    }

    pub fn clear(&mut self) {
        self.0.clear();
    }

    pub fn len(&mut self) -> usize {
        self.0.len()
    }

    pub fn is_valid_key(&self, key: &VecListKey) -> bool {
        self.0.is_valid_key(&key.0)
    }

    pub fn front(&self) -> Option<VecListKey> {
        self.0.front().map(|x| VecListKey(x))
    }

    pub fn back(&self) -> Option<VecListKey> {
        self.0.back().map(|x| VecListKey(x))
    }

    pub fn push_front(&mut self, item: T) -> VecListKey {
        VecListKey(self.0.push_front(item))
    }

    pub fn push_back(&mut self, item: T) -> VecListKey {
        VecListKey(self.0.push_back(item))
    }

    pub fn prev(&self, key: &VecListKey) -> Option<VecListKey> {
        self.0.prev(&key.0).map(|x| VecListKey(x))
    }

    pub fn next(&self, key: &VecListKey) -> Option<VecListKey> {
        self.0.next(&key.0).map(|x| VecListKey(x))
    }

    pub fn get(&self, key: &VecListKey) -> Option<&T> {
        self.0.get(&key.0)
    }

    pub fn get_mut(&mut self, key: &VecListKey) -> Option<&mut T> {
        self.0.get_mut(&key.0)
    }

    pub fn remove(&mut self, key: &VecListKey) -> Option<T> {
        self.0.remove(&key.0)
    }

    pub fn insert_before(&mut self, key: &VecListKey, item: T) -> Option<T> {
        self.0.insert_before(&key.0, item)
    }

    pub fn insert_after(&mut self, key: &VecListKey, item: T) -> Option<T> {
        self.0.insert_after(&key.0, item)
    }
}

impl<T> DoubleList<T> for VecDoubleList<T> {
    type Key = VecListKey;
    fn is_valid_key(&self, key: &Self::Key) -> bool {
        self.is_valid_key(key)
    }

    fn front(&self) -> Option<Self::Key> {
        self.front()
    }

    fn back(&self) -> Option<Self::Key> {
        self.back()
    }

    fn push_front(&mut self, item: T) -> Self::Key {
        self.push_front(item)
    }

    fn push_back(&mut self, item: T) -> Self::Key {
        self.push_back(item)
    }

    fn prev(&self, key: &Self::Key) -> Option<Self::Key> {
        self.prev(key)
    }

    fn next(&self, key: &Self::Key) -> Option<Self::Key> {
        self.next(key)
    }

    fn get(&self, key: &Self::Key) -> Option<&T> {
        self.get(key)
    }

    fn get_mut(&mut self, key: &Self::Key) -> Option<&mut T> {
        self.get_mut(key)
    }

    fn remove(&mut self, key: &Self::Key) -> Option<T> {
        self.remove(key)
    }

    fn insert_before(&mut self, key: &Self::Key, item: T) -> Option<T> {
        self.insert_before(key, item)
    }

    fn insert_after(&mut self, key: &Self::Key, item: T) -> Option<T> {
        self.insert_after(key, item)
    }
}


trait Storage<T> {
    type Key: Copy + Eq;
    fn clear(&mut self);
    fn len(&self) -> usize;
    fn get(&self, key: &Self::Key) -> Option<&T>;
    fn get_mut(&mut self, key: &Self::Key) -> Option<&mut T>;
    fn insert(&mut self, key: Self::Key, item: T);
    fn remove(&mut self, key: &Self::Key) -> Option<T>;
    fn contains_key(&self, key: &Self::Key) -> bool;
    fn get_new_key(&mut self) -> Self::Key;
}

#[derive(Clone, Debug)]
struct HashMapStorage<K, T> {
    hashmap: HashMap<K, T, nohash_hasher::BuildNoHashHasher<K>>,
    generation: usize
}

impl<T> Storage<T> for HashMapStorage<usize, T> {
    type Key = usize;
    fn clear(&mut self) { self.hashmap.clear() }
    fn len(&self) -> usize { self.hashmap.len() }
    fn get(&self, key: &Self::Key) -> Option<&T> { self.hashmap.get(key) }
    fn get_mut(&mut self, key: &Self::Key) -> Option<&mut T> { self.hashmap.get_mut(key) }
    fn insert(&mut self, key: Self::Key, item: T) { self.hashmap.insert(key, item); }
    fn remove(&mut self, key: &Self::Key) -> Option<T> { self.hashmap.remove(key) }
    fn contains_key(&self, key: &Self::Key) -> bool { self.hashmap.contains_key(key) }
    fn get_new_key(&mut self) -> Self::Key {
        if self.generation == usize::MAX {
            panic!("max key size usize max");
        }
        let key = self.generation;
        self.generation += 1;
        key
    }
}

struct VecStorage<T> {
    storage: Vec<T>,
    alive: Vec<u8>,
    freed: Vec<usize>,
}

impl<T> Storage<T> for VecStorage<T> {
    type Key = usize;

    fn clear(&mut self) {
        self.alive.clear();
        self.storage.clear();
        self.freed.clear();
    }

    fn len(&self) -> usize {
        self.storage.len() - self.freed.len()
    }

    fn get(&self, key: &Self::Key) -> Option<&T> {
        if self.contains_key(key) {
            return self.storage.get(*key);
        }
        None
    }

    fn get_mut(&mut self, key: &Self::Key) -> Option<&mut T> {
        if self.contains_key(key) {
            return self.storage.get_mut(*key);
        }
        None
    }

    fn insert(&mut self, key: Self::Key, item: T) {
        unsafe {
            let ptr = self.storage.as_mut_ptr().add(key);
            if self.contains_key(&key) {
                std::ptr::drop_in_place(ptr);
            } else {
                self.alive[key / 8] |= 1u8 << (key % 8);
            }
            std::ptr::write(ptr, item);
        }
    }

    fn remove(&mut self, key: &Self::Key) -> Option<T> {
        unsafe {
            let ptr = self.storage.as_mut_ptr().add(*key);
            if self.contains_key(key) {
                let item = std::ptr::read(ptr);
                self.alive[key / 8] &= !(1u8 << (key % 8));
                self.freed.push(*key);
                return Some(item);
            }
            None
        }
    }

    fn contains_key(&self, key: &Self::Key) -> bool {
        if *key >= self.storage.len() {
            return false;
        }
        let idx = key / 8;
        let bit = key % 8;
        self.alive[idx] & (1 << bit) != 0
    }

    fn get_new_key(&mut self) -> Self::Key {
        if self.freed.len() > 0 {
            return self.freed.pop().unwrap();
        }
        let idx = self.storage.len();
        if idx / 8 >= self.alive.len() {
            self.alive.push(0);
        }
        self.storage.reserve(idx + 1);
        unsafe {
            self.storage.set_len(idx + 1); 
        }
        idx
    }
}

impl<T> Drop for VecStorage<T> {
    fn drop(&mut self) {
        for i in 0..self.storage.len() {
            self.remove(&i);
        }
    }
}

impl<T> VecStorage<T> {
    fn new() -> Self {
        Self {
            storage: Vec::new(),
            alive: Vec::new(),
            freed: Vec::new(),
        }
    }
}

#[derive(Clone, Debug)]
struct GenericDoubleList<T, S: Storage<T, Key=usize>> {
    generation: usize,
    storage: S,
    front: usize,
    back: usize,
    _marker: PhantomData<T>
}

#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash)]
pub struct SListKey {
    generation: usize,
}

#[derive(Clone, Debug)]
struct GenericListItem<T> {
    item: T,
    prev: usize, // prev does not exist if prev == front
    next: usize, // next does not exist if next == back
}

impl<T, S> GenericDoubleList<GenericListItem<T>, S>
where S: Storage<GenericListItem<T>, Key=usize> {
    pub fn new(s: S) -> Self {
        Self {
            generation: 0,
            storage: s,
            front: usize::MAX,
            back: usize::MAX,
            _marker: PhantomData
        }
    }

    pub fn clear(&mut self) {
        self.generation = 0;
        self.storage.clear();
        self.front = usize::MAX;
        self.back = usize::MAX;
    }

    pub fn len(&mut self) -> usize {
        self.storage.len()
    }

    fn get_new_key(&mut self) -> SListKey {
        SListKey { generation: self.storage.get_new_key() }
    }

    fn is_front(&self, item: &GenericListItem<T>) -> bool {
        item.prev == usize::MAX
    }

    fn is_back(&self, item: &GenericListItem<T>) -> bool {
        item.next == usize::MAX
    }

    pub fn prev(&self, key: &SListKey) -> Option<SListKey> {
        if let Some(x) = self.storage.get(&key.generation) {
            if !self.is_front(x) {
                return Some(SListKey { generation: x.prev })
            }
        }
        None
    }
    pub fn next(&self, key: &SListKey) -> Option<SListKey> {
        if let Some(x) = self.storage.get(&key.generation) {
            if !self.is_back(x) {
                return Some(SListKey { generation: x.next })
            }
        }
        None
    }
    pub fn front(&self) -> Option<SListKey> {
        if self.storage.len() > 0 { // self.front is always valid if self.storage.len() > 0
            Some(SListKey { generation: self.front })
        } else {
            None
        }
    }

    pub fn back(&self) -> Option<SListKey> {
        if self.storage.len() > 0 { // self.back is always valid if self.storage.len() > 0
            Some(SListKey { generation: self.back })
        } else {
            None
        }
    }

    pub fn push_front(&mut self, item: T) -> SListKey { 
        let new_key = self.get_new_key();
        let mut item = GenericListItem {
            item,
            prev: usize::MAX,
            next: usize::MAX,
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

    pub fn push_back(&mut self, item: T) -> SListKey { 
        let new_key = self.get_new_key();
        let mut item = GenericListItem {
            item,
            prev: usize::MAX,
            next: usize::MAX,
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

    pub fn is_valid_key(&self, key: &SListKey) -> bool {
        self.storage.contains_key(&key.generation)
    }

    pub fn get(&self, key: &SListKey) -> Option<&T> {
        self.storage.get(&key.generation).map(|x| &x.item)
    }

    pub fn get_mut(&mut self, key: &SListKey) -> Option<&mut T> {
        self.storage.get_mut(&key.generation).map(|x| &mut x.item)
    }

    fn set_prev(&mut self, key: &SListKey, prev: usize) {
        if let Some(x) = self.storage.get_mut(&key.generation) {
            x.prev = prev;
        }
    }

    fn set_next(&mut self, key: &SListKey, next: usize) {
        if let Some(x) = self.storage.get_mut(&key.generation) {
            x.next = next;
        }
    }

    pub fn remove(&mut self, key: &SListKey) -> Option<T> {
        if let Some(x) = self.storage.remove(&key.generation) {
            self.set_next(&SListKey { generation: x.prev }, x.next);
            self.set_prev(&SListKey { generation: x.next }, x.prev);
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

    pub fn insert_before(&mut self, key: &SListKey, item: T) -> Option<T> {
        if self.storage.contains_key(&key.generation) {
            let new_key = self.get_new_key();
            let Some(cur) = self.storage.get_mut(&key.generation) else { unreachable!() };

            let mut item = GenericListItem { item, prev: usize::MAX, next: key.generation };
            let cur_prev = cur.prev;
            cur.prev = new_key.generation;
            if cur_prev == usize::MAX { // cur is front
                self.front = new_key.generation;
            } else {
                let Some(prev) = self.storage.get_mut(&cur_prev) else { unreachable!() };
                prev.next = new_key.generation;
                item.prev = cur_prev;
            }
            self.storage.insert(new_key.generation, item);
            return None;
        }
        Some(item)
    }

    pub fn insert_after(&mut self, key: &SListKey, item: T) -> Option<T> {
        if self.storage.contains_key(&key.generation) {
            let new_key = self.get_new_key();
            let Some(cur) = self.storage.get_mut(&key.generation) else { unreachable!() };
            
            let mut item = GenericListItem { item, prev: key.generation, next: usize::MAX };
            let cur_next = cur.next;
            cur.next = new_key.generation;
            if cur_next == usize::MAX { // cur is back
                self.back = new_key.generation;
            } else {
                let Some(next) = self.storage.get_mut(&cur_next) else { unreachable!() };
                next.prev = new_key.generation;
                item.next = cur_next;
            }
            self.storage.insert(new_key.generation, item);
            return None;
        }
        Some(item)
    }
}
