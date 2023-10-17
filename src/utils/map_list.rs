use std::collections::HashMap;

#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash)]
pub struct ListKeyMap {
    generation: usize,
}

#[derive(Clone, Debug)]
struct ListItem<T> {
    item: T,
    prev: usize, // prev does not exist if prev == front
    next: usize, // next does not exist if next == back
}

#[derive(Clone, Debug)]
pub struct HashMapDoubleList<T> {
    generation: usize,
    storage: HashMap<usize, ListItem<T>>,
    front: usize,
    back: usize,
}


impl<T> HashMapDoubleList<T> {
    pub fn front(&self) -> Option<ListKeyMap> {
        if self.storage.len() > 0 {
            Some(ListKeyMap { generation: self.front })
        } else {
            None
        }
    }

    pub fn back(&self) -> Option<ListKeyMap> {
        if self.storage.len() > 0 {
            Some(ListKeyMap { generation: self.back })
        } else {
            None
        }
    }

    pub fn push_front(&mut self, item: T) -> ListKeyMap { 
        let new_key = ListKeyMap { generation: self.generation };
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

    pub fn push_back(&mut self, item: T) -> ListKeyMap { 
        let new_key = ListKeyMap { generation: self.generation };
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

    pub fn is_valid_key(&self, key: &ListKeyMap) -> bool {
        self.storage.contains_key(&key.generation)
    }

    pub fn get(&self, key: &ListKeyMap) -> Option<&T> {
        self.storage.get(&key.generation).map(|x| &x.item)
    }

    pub fn get_mut(&mut self, key: &ListKeyMap) -> Option<&mut T> {
        self.storage.get_mut(&key.generation).map(|x| &mut x.item)
    }

    fn set_prev(&mut self, key: &ListKeyMap, prev: usize) {
        if let Some(x) = self.storage.get_mut(&key.generation) {
            x.prev = prev;
        }
    }

    fn set_next(&mut self, key: &ListKeyMap, next: usize) {
        if let Some(x) = self.storage.get_mut(&key.generation) {
            x.next = next;
        }
    }

    pub fn remove(&mut self, key: &ListKeyMap) -> Option<T> {
        if let Some(x) = self.storage.remove(&key.generation) {
            self.set_next(&ListKeyMap { generation: x.prev }, x.next);
            self.set_prev(&ListKeyMap { generation: x.next }, x.prev);
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

    pub fn insert_before(&mut self, key: &ListKeyMap, item: T) -> Option<T> {
        if let Some(cur) = self.storage.get_mut(&key.generation) {
            let new_key = ListKeyMap { generation: self.generation };
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

    pub fn insert_after(&mut self, key: &ListKeyMap, item: T) -> Option<T> {
        if let Some(cur) = self.storage.get_mut(&key.generation) {
            let new_key = ListKeyMap { generation: self.generation };
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