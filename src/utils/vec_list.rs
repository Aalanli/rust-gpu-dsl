
#[derive(Clone, Debug)]
struct FreedInfo {
    freed: Vec<usize>
}

impl FreedInfo {
    fn free(&mut self, pos: usize) {
        self.freed.push(pos);
    }

    fn alloc(&mut self) -> Option<usize>  {
        self.freed.pop()
    }
}


#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash)]
pub struct VecListKey {
    idx: usize
}

#[derive(Clone, Debug)]
struct ListItem<T> {
    item: T,
    prev: usize, // prev does not exist if prev == front
    next: usize, // next does not exist if next == back
}

#[derive(Clone, Debug)]
pub struct VecDoubleList<T> {
    freed: FreedInfo,
    storage: Vec<ListItem<T>>,
    front: usize,
    back: usize,
}

impl<T> VecDoubleList<T> {
    pub fn front(&self) -> Option<VecListKey> {
        todo!()
    }

    pub fn back(&self) -> Option<VecListKey> {
        todo!()
    }

    pub fn push_front(&mut self, item: T) -> VecListKey {
        todo!()
    }

    pub fn push_back(&mut self, item: T) -> VecListKey {
        todo!()
    }

    pub fn prev(&self, key: &VecListKey) -> Option<VecListKey> {
        todo!()
    }

    pub fn next(&self, key: &VecListKey) -> Option<VecListKey> {
        todo!()
    }

    pub fn get(&self, key: &VecListKey) -> Option<&T> {
        todo!()
    }

    pub fn get_mut(&mut self, key: &VecListKey) -> Option<&mut T> {
        todo!()
    }

    pub fn remove(&mut self, key: &VecListKey) -> Option<T> {
        todo!()
    }

    pub fn insert_before(&mut self, key: &VecListKey, item: T) -> Option<T> {
        todo!()
    }

    pub fn insert_after(&mut self, key: &VecListKey, item: T) -> Option<T> {
        todo!()
    }

    pub fn is_valid_key(&self, key: &VecListKey) -> bool {
        todo!()
    }

}