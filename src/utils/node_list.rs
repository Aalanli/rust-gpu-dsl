use std::marker::PhantomData;
use std::ptr::null;
use std::{ptr::NonNull, sync::atomic::AtomicUsize, hash::Hash};
use std::collections::HashSet;


struct ListNode<T, const N: usize> {
    items: NonNull<u8>,
    _marker: PhantomData<T>,
}

impl<T, const N: usize> ListNode<T, N> {
    unsafe fn ref_count(&mut self) -> usize {
        let ptr = self.items.as_ptr() as *mut usize;
        *ptr & !(1usize << 63)
    }

    unsafe fn inc_ref_count(&mut self) {
        let ptr = self.items.as_ptr() as *mut usize;
        let old = *ptr;
        assert!((old + 1) & (1usize << 63) == 0);
        *ptr = old + 1;
    }

    unsafe fn dec_ref_count(&mut self) {
        let ptr = self.items.as_ptr() as *mut usize;
        let old = *ptr;
        *ptr = old - 1;
    }

    unsafe fn is_parent_alive(&mut self) -> bool {
        let ptr = self.items.as_ptr() as *mut usize;
        *ptr & (1usize << 63) != 0
    }

    unsafe fn set_parent_alive(&mut self) {
        let ptr = self.items.as_ptr() as *mut usize;
        *ptr |= 1usize << 63;
    }

    unsafe fn set_parent_dead(&mut self) {
        let ptr = self.items.as_ptr() as *mut usize;
        *ptr &= !(1usize << 63);
    }

    unsafe fn prev(&mut self) -> Option<ListNode<T, N>> {
        let ptr = self.items.as_ptr() as *mut usize;
        if ptr.is_null() {
            None
        } else {
            let head = *(ptr as *mut *mut u8);
            Some(ListNode {
                items: NonNull::new(head).unwrap(),
                _marker: PhantomData,
            })
        }
    }

    unsafe fn next(&mut self) -> Option<ListNode<T, N>> {
        let ptr = self.items.as_ptr() as *mut usize;
        let ptr = ptr.add(2) as *mut u8;
        if ptr.is_null() {
            None
        } else {
            let head = *(ptr as *mut *mut u8);
            Some(ListNode {
                items: NonNull::new(head).unwrap(),
                _marker: PhantomData,
            })
        }
    }

    unsafe fn alive_buf(&mut self) -> *mut u8 {
        (self.items.as_ptr() as *mut usize).add(3) as *mut u8
    }

    unsafe fn is_alive(&mut self, index: usize) -> bool {
        // assert!(index < N);
        let ptr = self.alive_buf().add(index / 8);
        let mask = 1 << (index % 8);
        *ptr & mask != 0
    }

    unsafe fn start_of_buf(&mut self) -> *mut T {
        let n_alive = (N + 7) / 8;
        let alignment = std::mem::align_of::<T>();
        let bytes_offset = std::mem::size_of::<usize>() * 3 + n_alive;
        let offset = ((bytes_offset + alignment - 1) / alignment) * alignment;
        self.items.as_ptr().add(offset) as *mut T
    }

    unsafe fn new() -> Self {
        let n_alive = (N + 7) / 8;
        let alignment = std::mem::align_of::<T>();
        let bytes_offset = std::mem::size_of::<usize>() * 3 + n_alive;
        let offset = ((bytes_offset + alignment - 1) / alignment) * alignment;
        let size = offset + N * std::mem::size_of::<T>();
        let align = alignment.max(std::mem::align_of::<usize>());
        let layout = std::alloc::Layout::from_size_align(size, align).unwrap();
        let items = std::alloc::alloc(layout);

        Self { 
            items: NonNull::new(items).unwrap(),
            _marker: PhantomData,
        }
    }
}


pub struct NodeListKey<T, const N: usize> {
    node: ListNode<T, N>,
    index: usize,
}

impl<T, const N: usize> NodeListKey<T, N> {
    fn search_for_free_pos_prev(&mut self) -> Option<usize> {
        for i in (0..self.index-1).rev() {
            if unsafe { !self.node.is_alive(i) } {
                return Some(i);
            }
        }
        None
    }

    fn search_for_free_pos_after(&mut self) -> Option<usize> {
        for i in self.index+1..N {
            if unsafe { !self.node.is_alive(i) } {
                return Some(i);
            }
        }
        None
    }


}



#[test]
fn test() {
    println!("{}", isize::MAX as usize == !(1usize << 63));
}

// pub struct ListKeyNode<T> {
//     ptr: NonNull<NodeHelper<T>>,
// }

// impl<T> PartialEq for ListKeyNode<T> {
//     fn eq(&self, other: &Self) -> bool {
//         self.ptr == other.ptr
//     }
// }

// impl<T> Eq for ListKeyNode<T> {}

// impl<T> Hash for ListKeyNode<T> {
//     fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
//         self.ptr.hash(state);
//     }
// }

// impl<T> Drop for ListKeyNode<T> {
//     fn drop(&mut self) {
//         unsafe {
//             if (*self.ptr.as_ptr()).count.fetch_sub(1, std::sync::atomic::Ordering::Release) != 1 {
//                 return;
//             }

//             // (*self.ptr).count.load(std::sync::atomic::Ordering::Acquire);
//             std::sync::atomic::fence(std::sync::atomic::Ordering::Acquire);
//             std::alloc::dealloc(self.ptr.as_ptr() as *mut u8, std::alloc::Layout::new::<NodeHelper<T>>());
//         }
//     }
// }

// impl<T> Clone for ListKeyNode<T> {
//     fn clone(&self) -> Self {
//         unsafe {
//             let old = (*self.ptr.as_ptr()).count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
//             if old > isize::MAX as usize {
//                 panic!("too many clones");
//             }
//         }
//         ListKeyNode { ptr: self.ptr }
//     }
// }

// pub struct NodeDoubleList<T> {
//     alive: HashSet<ListKeyNode<T>>, // includes first and last
//     first: Option<ListKeyNode<T>>,
//     last: Option<ListKeyNode<T>>,
// }

// impl<T> Drop for NodeDoubleList<T> {
//     fn drop(&mut self) {
//         for key in self.alive.drain() {
//             unsafe {
//                 std::ptr::drop_in_place(&mut (*key.ptr.as_ptr()).item as *mut T);
//             }
//         }
//     }
// }

// impl<T> NodeDoubleList<T> {
//     fn new_node(item: T) -> ListKeyNode<T> {
//         let ptr = unsafe {
//             let ptr = std::alloc::alloc(std::alloc::Layout::new::<NodeHelper<T>>()) as *mut NodeHelper<T>;
//             (*ptr).prev = null();
//             (*ptr).next = null();
//             (*ptr).count.store(1, std::sync::atomic::Ordering::Relaxed);
//             (*ptr).item = item;
//             NonNull::new_unchecked(ptr)
//         };
//         ListKeyNode { ptr }
//     }
    
//     pub fn prev(&self, key: &ListKeyNode<T>) -> Option<ListKeyNode<T>> {
//         unsafe {
//             if (*key.ptr.as_ptr()).prev == null() {
//                 None
//             } else {
//                 let prev = (*key.ptr.as_ptr()).prev as *mut NodeHelper<T>;
//                 let old = (*prev).count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
//                 if old > isize::MAX as usize {
//                     panic!("too many clones");
//                 }
//                 Some(ListKeyNode { ptr: NonNull::new_unchecked(prev) })
//             }
//         }
//     }
//     pub fn next(&self, key: &ListKeyNode<T>) -> Option<ListKeyNode<T>> {
//         unsafe {
//             if (*key.ptr.as_ptr()).next == null() {
//                 None
//             } else {
//                 let next = (*key.ptr.as_ptr()).next as *mut NodeHelper<T>;
//                 let old = (*next).count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
//                 if old > isize::MAX as usize {
//                     panic!("too many clones");
//                 }
//                 Some(ListKeyNode { ptr: NonNull::new_unchecked(next) })
//             }
//         }
//     }
//     pub fn is_valid_key(&self, key: &ListKeyNode<T>) -> bool {
//         self.alive.contains(key)
//     }
//     pub fn front(&self) -> Option<ListKeyNode<T>> {
//         self.first.clone()
//     }
//     pub fn back(&self) -> Option<ListKeyNode<T>> {
//         self.last.clone()
//     }
//     pub fn push_front(&mut self, item: T) -> ListKeyNode<T> {
//         todo!()
//     }
//     // pub fn push_back(&mut self, item: T) -> ListKeyNode<T>;
//     // pub fn get(&self, key: &ListKeyNode<T>) -> Option<&T>;
//     // pub fn get_mut(&mut self, key: &ListKeyNode<T>) -> Option<&mut T>;
//     // pub fn remove(&mut self, key: &ListKeyNode<T>) -> Option<T>;
//     // pub fn insert_before(&mut self, key: &ListKeyNode<T>, item: T) -> Option<ListKeyNode<T>>;
//     // pub fn insert_after(&mut self, key: &ListKeyNode<T>, item: T) -> Option<ListKeyNode<T>>;

// }
