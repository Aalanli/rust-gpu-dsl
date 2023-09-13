
mod v1 {
    struct TCX;

    trait Query {
        type Q: Query;
        type A;
        type R;

        fn query(q: Self::Q, a: Self::A) -> Self::R;
    }


}


fn fail() {
    assert!(false);
}

fn a() {
    fail();
}

fn b() {
    a();
}

fn main() {
    b();
}

