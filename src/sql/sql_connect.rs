use polars::{prelude::*, docs::lazy};
use connectorx::prelude::*;
use std::{convert::TryFrom, clone};

struct MySql{
     db_user_name:String,
     db_pass_word:String,
     db_host:String,
     db_port:String,
     db_name:String,
}
pub fn main(){
    let mysql = MySql {
        db_user_name: String::from("hyuni"),
        db_pass_word: String::from("123412341234"),
        db_host: String::from("127.0.0.1"),
        db_port:String::from("3306"),
        db_name: String::from("Kesmai"),
    };
    let url= format!("mysql://{}:{}@{}:{}/{}?cxprotocol=binary",mysql.db_user_name,mysql.db_pass_word,mysql.db_host,mysql.db_port,mysql.db_name);

    let  source_conn = SourceConn::try_from(url.as_str()).unwrap();


    let queries = &[CXQuery::from("SELECT * FROM TB_SUBWAY_STATN_TK_GFF_TMP WHERE SUBWAY_STATN_NO < 100"), CXQuery::from("SELECT * FROM TB_SUBWAY_STATN_TK_GFF_TMP WHERE SUBWAY_STATN_NO >= 100")];
    let  destination = get_arrow2(&source_conn, None, queries).unwrap();
    let df = destination
    .polars()
    .expect("Should have worked")
    .sort(["SUBWAY_STATN_NO"], false,false)
    .unwrap();

    println!("{}",df);
}