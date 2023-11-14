use polars::prelude::*;


pub fn main(){
    let train_df = CsvReader::from_path("./datasets/titanic_beginner/train.csv")
    .unwrap()
    .finish()
    .unwrap().drop_many(&["PassengerId", "Name", "Ticket"]);
   let test_df = CsvReader::from_path("./datasets/titanic_beginner/test.csv")
   .unwrap()
   .finish()
   .unwrap().drop_many(&["Name", "Ticket"]);

   println!("{}",train_df.head(None));
  
   let value_counts = train_df.column("Pclass").unwrap().value_counts(false,false).unwrap();
   println!("{}",value_counts);


}