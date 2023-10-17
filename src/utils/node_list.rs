use std::ptr::null;
use std::{ptr::NonNull, sync::atomic::AtomicUsize, hash::Hash};
use std::collections::HashSet;

struct NodeHelper<T> {
    prev: *const NodeHelper<T>,
    next: *const NodeHelper<T>,
    count: AtomicUsize,
    item: T,
}

pub struct ListKeyNode<T> {
    ptr: NonNull<NodeHelper<T>>,
}

impl<T> PartialEq for ListKeyNode<T> {
    fn eq(&self, other: &Self) -> bool {
        self.ptr == other.ptr
    }
}

impl<T> Eq for ListKeyNode<T> {}

impl<T> Hash for ListKeyNode<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.ptr.hash(state);
    }
}

impl<T> Drop for ListKeyNode<T> {
    fn drop(&mut self) {
        unsafe {
            if (*self.ptr.as_ptr()).count.fetch_sub(1, std::sync::atomic::Ordering::Release) != 1 {
                return;
            }

            // (*self.ptr).count.load(std::sync::atomic::Ordering::Acquire);
            std::sync::atomic::fence(std::sync::atomic::Ordering::Acquire);
            std::alloc::dealloc(self.ptr.as_ptr() as *mut u8, std::alloc::Layout::new::<NodeHelper<T>>());
        }
    }
}

impl<T> Clone for ListKeyNode<T> {
    fn clone(&self) -> Self {
        unsafe {
            let old = (*self.ptr.as_ptr()).count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if old > isize::MAX as usize {
                panic!("too many clones");
            }
        }
        ListKeyNode { ptr: self.ptr }
    }
}

pub struct NodeDoubleList<T> {
    alive: HashSet<ListKeyNode<T>>, // includes first and last
    first: Option<ListKeyNode<T>>,
    last: Option<ListKeyNode<T>>,
}

impl<T> Drop for NodeDoubleList<T> {
    fn drop(&mut self) {
        for key in self.alive.drain() {
            unsafe {
                std::ptr::drop_in_place(&mut (*key.ptr.as_ptr()).item as *mut T);
            }
        }
    }
}

impl<T> NodeDoubleList<T> {
    fn new_node(item: T) -> ListKeyNode<T> {
        let ptr = unsafe {
            let ptr = std::alloc::alloc(std::alloc::Layout::new::<NodeHelper<T>>()) as *mut NodeHelper<T>;
            (*ptr).prev = null();
            (*ptr).next = null();
            (*ptr).count.store(1, std::sync::atomic::Ordering::Relaxed);
            (*ptr).item = item;
            NonNull::new_unchecked(ptr)
        };
        ListKeyNode { ptr }
    }
    
    pub fn prev(&self, key: &ListKeyNode<T>) -> Option<ListKeyNode<T>> {
        unsafe {
            if (*key.ptr.as_ptr()).prev == null() {
                None
            } else {
                let prev = (*key.ptr.as_ptr()).prev as *mut NodeHelper<T>;
                let old = (*prev).count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if old > isize::MAX as usize {
                    panic!("too many clones");
                }
                Some(ListKeyNode { ptr: NonNull::new_unchecked(prev) })
            }
        }
    }
    pub fn next(&self, key: &ListKeyNode<T>) -> Option<ListKeyNode<T>> {
        unsafe {
            if (*key.ptr.as_ptr()).next == null() {
                None
            } else {
                let next = (*key.ptr.as_ptr()).next as *mut NodeHelper<T>;
                let old = (*next).count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if old > isize::MAX as usize {
                    panic!("too many clones");
                }
                Some(ListKeyNode { ptr: NonNull::new_unchecked(next) })
            }
        }
    }
    pub fn is_valid_key(&self, key: &ListKeyNode<T>) -> bool {
        self.alive.contains(key)
    }
    pub fn front(&self) -> Option<ListKeyNode<T>> {
        self.first.clone()
    }
    pub fn back(&self) -> Option<ListKeyNode<T>> {
        self.last.clone()
    }
    pub fn push_front(&mut self, item: T) -> ListKeyNode<T> {
        todo!()
    }
    // pub fn push_back(&mut self, item: T) -> ListKeyNode<T>;
    // pub fn get(&self, key: &ListKeyNode<T>) -> Option<&T>;
    // pub fn get_mut(&mut self, key: &ListKeyNode<T>) -> Option<&mut T>;
    // pub fn remove(&mut self, key: &ListKeyNode<T>) -> Option<T>;
    // pub fn insert_before(&mut self, key: &ListKeyNode<T>, item: T) -> Option<ListKeyNode<T>>;
    // pub fn insert_after(&mut self, key: &ListKeyNode<T>, item: T) -> Option<ListKeyNode<T>>;

}
