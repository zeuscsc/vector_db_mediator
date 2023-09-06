from pymilvus import Collection as MilvusCollection
from pymilvus import CollectionSchema, FieldSchema, DataType
from llm_mediator.llm import LLM
from llm_mediator.embedding import Embedding
from llm_mediator.gpt import GPT

DEFAULT_LOCAL_EMBEDDING_INDEX_PARAMS={"metric_type":"IP","index_type":"IVF_FLAT","params":{"nlist":1024}}
DEFAULT_LOCAL_EMBEDDING_SEARCH_PARAMS={"metric_type": "IP", "params": {"nprobe": 10}}
DEFAULT_GPT_EMBEDDING_INDEX_PARAMS = {'index_type': 'IVF_FLAT','metric_type': 'L2','params': {'nlist': 1024}}
DEFAULT_GPT_EMBEDDING_SEARCH_PARAMS={"metric_type": "L2"}

class FieldSchemaHelper(FieldSchema):
    def __init__(self,model:LLM,name,dtype,description: str = "",**kwargs):
        if dtype == DataType.FLOAT_VECTOR and kwargs.get("dim") is None:
            kwargs["dim"] = model.model_class.embedding_size
        if dtype == DataType.VARCHAR:
            kwargs["max_length"]=65535
        super().__init__(name=name,dtype=dtype,description=description,**kwargs)
    pass

class MilvusMediator:
    def __init__(self,db_name,alias,index_params = DEFAULT_LOCAL_EMBEDDING_INDEX_PARAMS,llm_type=None):
        self.llm_type = llm_type
        if llm_type is not None:
            if llm_type == Embedding:
                index_params = DEFAULT_LOCAL_EMBEDDING_INDEX_PARAMS
            elif llm_type == GPT:
                index_params = DEFAULT_GPT_EMBEDDING_INDEX_PARAMS
        self.db_name = db_name
        self.alias = alias
        self.index_params = index_params
        self.collections:dict[str,MilvusCollection]=dict()
        self.current_collection:MilvusCollection=None
        self.connect_to_milvusdb()
    
    def connect_to_milvusdb(self):
        from pymilvus import connections
        from pymilvus import db
        try:
            host = "localhost"
            port = 19530
            connections.add_connection(default={"host": host, "port": port})
            connections.connect(alias=self.alias)
        except Exception as e:
            print(e)
            if self.db_name not in db.list_database():
                db.create_database(self.db_name)
    def switch_collection(self,collection_name):
        self.current_collection_name = collection_name
        self.current_collection = self.collections[collection_name]
        self.current_collection.load()
    def connect_to_milvus_collection(self,collection_name)->MilvusCollection:
        self.current_collection_name = collection_name
        self.collections[collection_name]= MilvusCollection(collection_name,using=self.alias)
        self.current_collection = self.collections[collection_name]
        self.collections[collection_name].load()
    def initialize_schema(self,collection_name,fields:list[FieldSchema])->MilvusCollection:
        self.current_collection_name = collection_name
        from pymilvus import utility
        if utility.has_collection(self.current_collection_name,using=self.alias):
            utility.drop_collection(self.current_collection_name)
        
        schema = CollectionSchema(fields=fields)
        collection = MilvusCollection(name=self.current_collection_name, schema=schema,using=self.alias)
        for field in fields:
            if field.dtype == DataType.FLOAT_VECTOR:
                collection.create_index(field_name=field.name, index_params=self.index_params)
        self.collections[collection_name] = collection
        self.current_collection = collection
        return collection
    def insert(self,data:list|dict):
        self.current_collection.insert(data)
    def search(self,limit=1,anns_field="embedding",param=DEFAULT_LOCAL_EMBEDDING_SEARCH_PARAMS,**kwargs):
        if self.llm_type is not None:
            if self.llm_type == Embedding:
                param = DEFAULT_LOCAL_EMBEDDING_SEARCH_PARAMS
            elif self.llm_type == GPT:
                param = DEFAULT_GPT_EMBEDDING_SEARCH_PARAMS
        kwargs["param"] = param
        kwargs["limit"] = limit
        kwargs["anns_field"] = anns_field
        return self.current_collection.search(**kwargs)
    pass