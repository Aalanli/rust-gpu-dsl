
// todo impl drop
// pub struct DoublyLinkedList<T> {
//     alive: Vec<bool>,
//     storage: Vec<ListItem<T>>,
//     num_empty: usize,
//     first_last: Option<(usize, usize)>,
// }

// impl<T> DoublyLinkedList<T> {
//     const SEARCH_SIZE: usize = 4;
//     pub fn new() -> Self {
//         DoublyLinkedList {
//             alive: vec![],
//             storage: vec![],
//             num_empty: 0,
//             first_last: None,
//         }
//     }

//     fn is_valid_idx(&self, idx: usize) -> bool {
//         self.alive.get(idx) == Some(&true)
//     }

//     pub fn is_valid_key(&self, key: &ListKey) -> bool {
//         self.is_valid_idx(key.idx)
//     }

//     pub fn prev(&self, key: &ListKey) -> Option<ListKey> {
//         if self.is_valid_key(key) {
//             let item = unsafe { self.storage.get_unchecked(key.idx) };
//             if item.prev != key.idx {
//                 Some(ListKey { idx: item.prev })
//             } else {
//                 None
//             }
//         } else {
//             None
//         }
//     }

//     pub fn next(&self, key: &ListKey) -> Option<ListKey> {
//         if self.is_valid_key(key) {
//             let item = unsafe { self.storage.get_unchecked(key.idx) };
//             if item.next != key.idx {
//                 Some(ListKey { idx: item.next })
//             } else {
//                 None
//             }
//         } else {
//             None
//         }
//     }

//     pub fn get(&self, key: &ListKey) -> Option<&T> {
//         if self.is_valid_key(key) {
//             let item = unsafe { self.storage.get_unchecked(key.idx) };
//             Some(&item.item)
//         } else {
//             None
//         }
//     }

//     pub fn get_mut(&mut self, key: &ListKey) -> Option<&mut T> {
//         if self.is_valid_key(key) {
//             let item = unsafe { self.storage.get_unchecked_mut(key.idx) };
//             Some(&mut item.item)
//         } else {
//             None
//         }
//     }

//     pub fn remove(&mut self, key: &ListKey) -> Option<T> {
//         if self.is_valid_key(key) {
//             unsafe {
//                 *self.alive.get_unchecked_mut(key.idx) = false;
//                 self.num_empty += 1;
//                 let item = self.storage.as_mut_ptr().add(key.idx).read();
//                 if item.prev != key.idx {
//                     if self.is_valid_idx(item.prev) {
//                         let prev_item = self.storage.as_mut_ptr().add(item.prev);
//                         (*prev_item).next = item.next;
//                     }
//                 } else { // first item
//                     self.first_last = self.first_last.map(|(_, lst)| (item.next, lst));
//                 }
//                 if item.next != key.idx {
//                     if self.is_valid_idx(item.next) {
//                         let next_item = self.storage.as_mut_ptr().add(item.next);
//                         (*next_item).prev = item.prev;
//                     }
//                 } else { // last item
//                     self.first_last = self.first_last.map(|(fst, _)| (fst, item.prev));
//                 }
//                 Some(item.item)
//             }
//         } else {
//             None
//         }
//     }

//     fn get_empty_spot(&mut self, key_idx: usize) -> usize {
//         unsafe {
//             for i in (0..Self::SEARCH_SIZE).rev() {
//                 if key_idx >= i {
//                     let idx = key_idx - i;
//                     if *self.alive.get_unchecked(idx) == true {
//                         return idx;
//                     }
//                 } else {
//                     break;
//                 }
//             }
//             for i in 0..Self::SEARCH_SIZE {
//                 let idx = key_idx + i;
//                 if idx < self.alive.len() {
//                     if *self.alive.get_unchecked(idx) == true {
//                         return idx;
//                     }
//                 } else {
//                     break;
//                 }
//             }
//             self.storage.len()
//         }
//     }

//     unsafe fn set_pos(&mut self, idx: usize, item: ListItem<T>) {
//         if idx == self.storage.len() {
//             self.storage.push(item);
//             self.alive.push(true);
//         } else {
//             *self.storage.get_unchecked_mut(idx) = item;
//             *self.alive.get_unchecked_mut(idx) = true;
//         }
//     }

//     pub fn insert_before(&mut self, key: &ListKey, item: T) -> Option<ListKey> {
//         if self.is_valid_key(key) {
//             unsafe {
//                 let item = ListItem {
//                     item,
//                     prev: self.storage.get_unchecked(key.idx).prev,
//                     next: key.idx,
//                 };
//                 let empty_spot = self.get_empty_spot(key.idx);

//                 if key.idx == self.first_last.unwrap().0 { // insert before first
//                     self.first_last = self.first_last.map(|(_, lst)| (empty_spot, lst));
//                 }

//                 self.storage.get_unchecked_mut(key.idx).prev = empty_spot;
//                 self.set_pos(empty_spot, item);
//                 Some(ListKey { idx: empty_spot })
//             }
//         } else {
//             None
//         }
//     }

//     pub fn insert_after(&mut self, key: &ListKey, item: T) -> Option<ListKey> {
//         if self.is_valid_key(key) {
//             unsafe {
//                 let item = ListItem {
//                     item,
//                     prev: key.idx,
//                     next: self.storage.get_unchecked(key.idx).next,
//                 };

//                 let empty_spot = self.get_empty_spot(key.idx);
//                 if key.idx == self.first_last.unwrap().1 {// insert after last 
//                     self.first_last = self.first_last.map(|(fst, _)| (fst, empty_spot));
//                 }
//                 self.storage.get_unchecked_mut(key.idx).next = empty_spot;
//                 self.set_pos(empty_spot, item);
//                 Some(ListKey { idx: empty_spot })
//             }
//         } else {
//             None
//         }
//     }

//     pub fn front(&self) -> Option<ListKey> {
//         self.first_last.map(|(fst, _)| ListKey { idx: fst })
//     }

//     pub fn back(&self) -> Option<ListKey> {
//         self.first_last.map(|(_, lst)| ListKey { idx: lst })
//     }

//     pub fn push_back(&mut self, item: T) -> ListKey {
//         let (first, last) = self.first_last.unwrap_or((0, 0));
//         let prev_node = self.prev(&ListKey { idx: last })
//             .unwrap_or(ListKey { idx: 0 });
//         let empty_spot = self.get_empty_spot(last);
//         let item = ListItem {
//             item,
//             prev: prev_node.idx,
//             next: empty_spot,
//         };
//         unsafe {
//             self.storage.get_unchecked_mut(last).next = empty_spot;
//             self.set_pos(empty_spot, item);
//         }
//         self.first_last = Some((first, empty_spot));
//         ListKey { idx: empty_spot }
//     }

//     pub fn push_front(&mut self, item: T) -> ListKey {
//         let (first, last) = self.first_last.unwrap_or((0, 0));
//         let next_node = self.next(&ListKey { idx: first })
//             .unwrap_or(ListKey { idx: 0 });
//         let empty_spot = self.get_empty_spot(first);
//         let item = ListItem {
//             item,
//             prev: empty_spot,
//             next: next_node.idx,
//         };
//         unsafe {
//             self.storage.get_unchecked_mut(first).prev = empty_spot;
//             self.set_pos(empty_spot, item);
//         }
//         self.first_last = Some((empty_spot, last));
//         ListKey { idx: empty_spot }
//     }


// }

// #[derive(PartialEq, Eq, Clone, Copy, Debug, Hash)]
// pub struct ListKey {
//     idx: usize
// }



// #[derive(Clone, Debug)]
// pub struct VecDoubleList<T> {
//     freed: FreedInfo,
//     storage: Vec<ListItem<T>>,
//     front: usize,
//     back: usize,
//     global_offset: usize
// }


// #[derive(Clone, Debug)]
// struct FreedInfo {
//     freed: Vec<usize>
// }

// use std::hash::Hash;
// use std::ptr::{NonNull, null};
// use std::sync::Arc;
// use std::sync::atomic::AtomicUsize;
// impl FreedInfo {
//     fn free(&mut self, pos: usize) {
//         self.freed.push(pos);
//     }

//     fn alloc(&mut self) -> Option<usize>  {
//         self.freed.pop()
//     }
// }

// #[derive(PartialEq, Eq, Clone, Copy, Debug, Hash)]
// pub struct ListKeyVec {
//     offset: usize
// }
