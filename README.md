# MMDT_i2t_data
The code to upload MMDT's data onto huggingface  
Contain 3 perspectives: adversarial robustness, fairness, privacy.
# Key of each perspective
## adv:
origin_attribute: the original attribute of data in split "attribute"  
origin_object: the original object of data in split "object"   
origin_relation: the original relation of data in split "spatial"   
object_a, object_b: the objects used in split "spatial", template is \{object_a\} \{relation\} \{object_b\}  
object: the object used in split "attribute", template is \{attribute\} \{object\}
label: the label of the data
surrogate_model: the model attacked by the algorithm 
algorithm: the algorithm used to generate this data

## fairness:
q_gender: questions about gender
q_race: questions about race
q_age: questions about age

## privacy:
task: street_view or selfies
type_street_view: the specific difficulty of the street_view task, single/group & text/no text
country, state_province, city, latitude, longitude, zipcode: the label of data in split "street_view"
ethnicity: caucasians or hispanics, only works in split "selfies"
label_selfies: the label of data in split "selfies"
type_selfies: ID or Selfie, distinguish the type of image in split "selfies"