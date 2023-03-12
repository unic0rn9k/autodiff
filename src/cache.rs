use std::{
    collections::HashMap,
    mem::{size_of, transmute},
};

use crate::NodeId;

pub struct Cache {
    data: Vec<u8>,
    entries: HashMap<NodeId, *const ()>,
}

impl Cache {
    pub fn get<'a, T>(&'a self, id: &NodeId) -> Option<&'a T> {
        self.entries.get(id).map(|n| unsafe { transmute(n) })
    }

    /// Unlike HashMap, this returns a reference to the newly inserted item.
    pub fn insert<'a, T: Sized>(&'a mut self, id: NodeId, v: T) -> &'a T {
        let len = self.data.len();
        self.data.resize(size_of::<T>(), 0);
        let v_ref = unsafe { transmute::<_, &mut T>(&mut self.data[len]) };
        *v_ref = v;
        self.entries.insert(id, v_ref as *mut _ as *const ());
        v_ref
    }
}
