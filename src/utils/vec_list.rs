use std::{hash::Hash, marker::PhantomData};
use rand::{thread_rng, Rng};

use super::{DoubleList, DoubleListOwnedIter, DoubleListRefIter};

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct VecListKey {
    index: usize,
    generation: usize,
    uid: usize
}

pub struct VecDoubleList<T> {
    storage: Vec<GenericListItem<T>>,
    freed: Vec<usize>,
    generation: Vec<usize>,
    front: usize,
    back: usize,
    uid: usize,
}

struct GenericListItem<T> {
    item: T,
    prev: usize,
    next: usize,
}

impl<T> VecDoubleList<T> {
    const SENTINEL: usize = usize::MAX; // represents the front or end of the list
    const EMPTY_MARKER: usize = 1 << 63; // represents that the position is empty

    pub fn new() -> Self {
        let mut rng = thread_rng();
        let uid = rng.gen();
        Self {
            storage: Vec::new(),
            freed: Vec::new(),
            generation: Vec::new(),
            front: Self::SENTINEL,
            back: Self::SENTINEL,
            uid,
        }
    }

    /// A valid key is a key that is spawned from the same VecDoubleList instance, and that has not ever been removed
    pub fn is_valid_key(&self, key: &VecListKey) -> bool {
        // if the generation has 1 set as the first bit, then its already empty
        // valid generations are always less than 1 << 63
        key.index < self.generation.len() && unsafe { *self.generation.get_unchecked(key.index) } == key.generation && key.uid == self.uid
    }
    
    pub fn clear(&mut self) {
        for i in 0..self.generation.len() {
            unsafe { 
                if (*self.generation.get_unchecked(i) & Self::EMPTY_MARKER) == 0 {
                    let ptr = &mut self.storage.get_unchecked_mut(i).item as *mut T;
                    std::ptr::drop_in_place(ptr);
                    *self.generation.get_unchecked_mut(i) |= Self::EMPTY_MARKER;
                }
            }
        }
        unsafe {
            self.storage.set_len(0);
        }
        self.freed.clear();
        self.front = Self::SENTINEL;
        self.back = Self::SENTINEL;
    }

    pub fn len(&self) -> usize {
        self.storage.len() - self.freed.len()
    }

    pub fn front(&self) -> Option<VecListKey> {
        if self.front == Self::SENTINEL {
            None
        } else { // assumes that if the front is not SENTINEL, then it is a valid position
            debug_assert!(unsafe { self.generation(self.front) } & Self::EMPTY_MARKER == 0);
            Some(VecListKey { index: self.front, generation: unsafe { self.generation(self.front) }, uid: self.uid })
        }
    }

    pub fn back(&self) -> Option<VecListKey> {
        if self.back == Self::SENTINEL {
            None
        } else { // assumes that if the back is not SENTINEL, then it is a valid position
            debug_assert!(unsafe { self.generation(self.back)} & Self::EMPTY_MARKER == 0);
            Some(VecListKey { index: self.back, generation: unsafe { self.generation(self.back) }, uid: self.uid })
        }
    }

    unsafe fn prev_index(&self, key: usize) -> usize {
        self.storage.get_unchecked(key).prev
    }

    unsafe fn next_index(&self, key: usize) -> usize {
        self.storage.get_unchecked(key).next
    }

    unsafe fn generation(&self, key: usize) -> usize {
        debug_assert!(key < self.generation.len());
        *self.generation.get_unchecked(key)
    }

    unsafe fn generation_mut(&mut self, key: usize) -> &mut usize {
        debug_assert!(key < self.generation.len());
        self.generation.get_unchecked_mut(key)
    }

    pub fn prev(&self, key: &VecListKey) -> Option<VecListKey> {
        if !self.is_valid_key(key) {
            None
        } else {
            let prev = unsafe { self.prev_index(key.index) };
            if prev == Self::SENTINEL {
                None
            } else {
                debug_assert!(unsafe { self.generation(prev) } & Self::EMPTY_MARKER == 0);
                Some(VecListKey { index: prev, generation: unsafe { self.generation(prev) }, uid: self.uid })
            }
        }
    }

    pub fn next(&self, key: &VecListKey) -> Option<VecListKey> {
        if !self.is_valid_key(key) {
            None
        } else {
            let next = unsafe { self.next_index(key.index) };
            if next == Self::SENTINEL {
                None
            } else {
                debug_assert!(unsafe { self.generation(next) } & Self::EMPTY_MARKER == 0);
                Some(VecListKey { index: next, generation: unsafe { self.generation(next) }, uid: self.uid })
            }
        }
    }

    pub fn get(&self, key: &VecListKey) -> Option<&T> {
        if !self.is_valid_key(key) {
            None
        } else {
            debug_assert!(unsafe { self.generation(key.index) } & Self::EMPTY_MARKER == 0);
            Some(unsafe { &self.storage.get_unchecked(key.index).item })
        }
    }

    pub fn get_mut(&mut self, key: &VecListKey) -> Option<&mut T> {
        if !self.is_valid_key(key) {
            None
        } else {
            debug_assert!(unsafe { self.generation(key.index) } & Self::EMPTY_MARKER == 0);
            Some(unsafe { &mut self.storage.get_unchecked_mut(key.index).item })
        }
    }

    unsafe fn get_new_key(&mut self) -> VecListKey {
        if let Some(x) = self.freed.pop() {
            let gen = self.generation_mut(x);
            debug_assert!(*gen & Self::EMPTY_MARKER != 0); // assert that the generation is empty
            if *gen & !Self::EMPTY_MARKER == !Self::EMPTY_MARKER { // 0111111111111...
                panic!("generation overflow");
            }
            *gen &= !Self::EMPTY_MARKER; // mark as not empty
            *gen += 1;
            let generation = *gen;
            self.set_prev(x, Self::SENTINEL);
            self.set_next(x, Self::SENTINEL);
            VecListKey { index: x, generation, uid: self.uid }
        } else {
            debug_assert!(self.storage.len() <= self.generation.len());
            let generation = if self.storage.len() == self.generation.len() {
                self.generation.push(0);
                0
            } else {
                let gen = self.generation.get_unchecked_mut(self.storage.len());
                debug_assert!(*gen & Self::EMPTY_MARKER != 0); // assert that the generation is empty
                if *gen & !Self::EMPTY_MARKER == !Self::EMPTY_MARKER {
                    panic!("generation overflow");
                }
                *gen = *gen & !Self::EMPTY_MARKER; // mark as not empty
                *gen += 1;
                *gen
            };
            self.storage.push(GenericListItem { item: std::mem::zeroed(), prev: Self::SENTINEL, next: Self::SENTINEL });
            VecListKey { index: self.storage.len() - 1, generation, uid: self.uid }
        }
    }

    unsafe fn set_item(&mut self, key: usize, item: T) {
        let ptr = self.storage.get_unchecked_mut(key);
        std::ptr::write(&mut ptr.item as *mut T, item);
    }

    unsafe fn set_prev(&mut self, key: usize, index: usize) {
        let ptr = self.storage.get_unchecked_mut(key);
        ptr.prev = index;
    }

    unsafe fn set_next(&mut self, key: usize, index: usize) {
        let ptr = self.storage.get_unchecked_mut(key);
        ptr.next = index;
    }

    pub fn push_front(&mut self, item: T) -> VecListKey {
        unsafe {
            let key = self.get_new_key();
            self.set_item(key.index, item);
            if self.front == Self::SENTINEL { // if front is sentinel then back is also sentinel
                self.back = key.index;
            } else {
                self.set_next(key.index, self.front);
                self.set_prev(self.front, key.index);
            }
            self.front = key.index;
            key
        }
    }

    pub fn push_back(&mut self, item: T) -> VecListKey {
        unsafe {
            let key = self.get_new_key();
            self.set_item(key.index, item);
            if self.front == Self::SENTINEL { // if front is sentinel then back is also sentinel
                self.front = key.index;
            } else {
                self.set_prev(key.index, self.back);
                self.set_next(self.back, key.index);
            }
            self.back = key.index;
            key
        }
    }

    // Option[prev] <-> key
    // =>
    // Option[prev] <-> new_key <-> key
    pub fn insert_before(&mut self, key: &VecListKey, item: T) -> Option<T> {
        if self.is_valid_key(key) {
            unsafe {
                let new_key = self.get_new_key();
                self.set_item(new_key.index, item);
                self.set_next(new_key.index, key.index);
                if let Some(prev) = self.prev(key) {
                    self.set_next(prev.index, new_key.index);
                    self.set_prev(new_key.index, prev.index);
                } else {
                    // self.set_prev(new_key.index, Self::SENTINEL);
                    self.front = new_key.index;
                }
                self.set_prev(key.index, new_key.index);
            }
            return None;
        }
        Some(item)
    }


    // key <-> Option[next]
    // =>
    // key <-> new_key <-> Option[next]
    pub fn insert_after(&mut self, key: &VecListKey, item: T) -> Option<T> {
        if self.is_valid_key(key) {
            unsafe {
                let new_key = self.get_new_key();
                self.set_item(new_key.index, item);
                self.set_prev(new_key.index, key.index);
                if let Some(next) = self.next(key) {
                    self.set_prev(next.index, new_key.index);
                    self.set_next(new_key.index, next.index);
                } else {
                    // self.set_next(new_key.index, Self::SENTINEL);
                    self.back = new_key.index;
                }
                self.set_next(key.index, new_key.index);
            }
            return None;
        }
        Some(item)
    }

    pub fn remove(&mut self, key: &VecListKey) -> Option<T> {
        if self.is_valid_key(key) {
            unsafe {
                debug_assert!(self.generation(key.index) & Self::EMPTY_MARKER == 0);
                let item = std::ptr::read(&self.storage.get_unchecked(key.index).item as *const T);
                *self.generation_mut(key.index) |= Self::EMPTY_MARKER;
                self.freed.push(key.index);
                let prev = self.prev_index(key.index);
                let next = self.next_index(key.index);
                if prev != Self::SENTINEL {
                    self.set_next(prev, next);
                } else {
                    self.front = next;
                }

                if next != Self::SENTINEL {
                    self.set_prev(next, prev);
                } else {
                    self.back = prev;
                }

                return Some(item);
            }
        }
        None
    }

    pub fn iter(&self) -> DoubleListRefIter<T, Self> {
        DoubleListRefIter { list: self, key: self.front(), _marker: PhantomData }
    }

    pub fn iter_mut(&mut self) -> DoubleListMutIter<T> {
        let front = self.front();
        DoubleListMutIter { iter: self, key: front, }
    }
}

impl<T> Drop for VecDoubleList<T> {
    fn drop(&mut self) {
        self.clear();
    }
}

impl<T: Clone> Clone for VecDoubleList<T> {
    fn clone(&self) -> Self {
        let freed = self.freed.clone();
        let mut generation = vec![0; self.generation.len()];
        let mut storage = Vec::new();
        storage.reserve(self.storage.len());
        for i in 0..self.storage.len() {
            unsafe {
                if self.generation(i) & Self::EMPTY_MARKER == 0 {
                    storage.push(GenericListItem { item: self.storage.get_unchecked(i).item.clone(), prev: self.storage.get_unchecked(i).prev, next: self.storage.get_unchecked(i).next });
                } else {
                    *generation.get_unchecked_mut(i) |= Self::EMPTY_MARKER;
                    storage.push(GenericListItem { item: std::mem::zeroed(), prev: Self::SENTINEL, next: Self::SENTINEL })
                }
            }
        }
        let mut rng = thread_rng();
        VecDoubleList { storage, freed, generation, front: self.front, back: self.back, uid: rng.gen() }
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

impl<T> IntoIterator for VecDoubleList<T> {
    type IntoIter = DoubleListOwnedIter<T, Self>;
    type Item = T;
    fn into_iter(self) -> Self::IntoIter {
        DoubleListOwnedIter { list: self, _marker: PhantomData }
    }
}

pub struct DoubleListMutIter<'a, T> {
    iter: &'a mut VecDoubleList<T>,
    key: Option<VecListKey>,
}

impl<'a, T> Iterator for DoubleListMutIter<'a, T> {
    type Item = &'a mut T;
    fn next(&mut self) -> Option<Self::Item> {
        let key = self.key.as_ref()?;
        if self.iter.is_valid_key(key) {
            let idx = key.index;
            self.key = self.iter.next(key);
            unsafe {
                return Some(&mut (*self.iter.storage.as_mut_ptr().add(idx)).item);
            }
        }
        None
    }
}

#[test]
fn test_vec_list() {
    let mut list = VecDoubleList::<usize>::new();
    let k1 = list.push_front(0);
    list.insert_after(&k1, 1);
    let k2 = list.next(&k1).unwrap();
    list.insert_after(&k2, 2);
    let k3 = list.next(&k2).unwrap();
    list.insert_after(&k3, 3);
    let items = list.iter().cloned().collect::<Vec<_>>();
    // println!("{:?}", items);
    assert!(items == vec![0, 1, 2, 3]);
    list.remove(&k2);
    let items = list.iter().cloned().collect::<Vec<_>>();
    // println!("{:?}", items);
    assert!(items == vec![0, 2, 3]);
    let items2 = list.clone().into_iter().collect::<Vec<_>>();
    println!("{:?}", items2);
    assert!(items2 == vec![0, 2, 3]);
    let mut list2 = VecDoubleList::<usize>::new();
    let p1 = list2.push_front(0);
    assert!(list2.is_valid_key(&k2) == false);
    assert!(list2.is_valid_key(&p1) == true);
    assert!(list.is_valid_key(&k2) == false);

    let prev = list.prev(&k3).unwrap();
    assert!(list.is_valid_key(&prev) == true);
    assert!(list.get(&prev) == Some(&0));

}