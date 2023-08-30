# Science

Identifying vehicles and understanding variation of them according to date and time tells many things. For example, campervan and RVs entering an entrance of a preserve may indicate the level of popularity or crowdedness. It is important to understand this in terms of this information can drive relative environment research or regularions.

# AI at Edge

The code runs the YOLOv7 model which is trained with COCO dataset with a given time interval. In each run, it takes a still image from a given camera (bottom) and outputs types of any recognized vehicles listed in `coco.names`. The model resizes the input image to 640x640 (WxH) as the model was trained with the size.

This model will be updated with internal dataset for identifying smaller passenger vehicles (car and small SUV), larger passenger vehicles (large SUVs and vans), RVs and campers, boat trailers, and busses and trucks.

# Ontology

The code publishes measurement with topic `env.vehicle.type`. Value for a topic indicates the type of the object recognized.

# Inference from Sage codes
To query the output from the plugin, you can do with python library 'sage_data_client':
```
import sage_data_client

# query and load data into pandas data frame
df = sage_data_client.query(
    start="-1h",
    filter={
        "name": "env.vehicle.type",
    }
)

# print results in data frame
print(df)
# print results by its name
print(df.name.value_counts())
# print filter names
print(df.name.unique())
```
For more information, please see [Access and use data documentation](https://docs.sagecontinuum.org/docs/tutorials/accessing-data) and [sage_data_client](https://pypi.org/project/sage-data-client/).
