Search.setIndex({docnames:["experiments/index","getting-started/complete-tour","getting-started/index","getting-started/installation","getting-started/resources","index","reference/geoparser/index","reference/geoparser/linker","reference/geoparser/pipeline","reference/geoparser/ranker","reference/geoparser/recogniser","reference/index","reference/utils/deezy_processing","reference/utils/get_data","reference/utils/index","reference/utils/ner","reference/utils/preprocess_data","reference/utils/process_data","reference/utils/process_wikipedia","reference/utils/rel/entity_disambiguation","reference/utils/rel/index","reference/utils/rel/mulrel_ranker","reference/utils/rel/utils","reference/utils/rel/vocabulary","reference/utils/rel_e2e","reference/utils/rel_utils","t-res-api/index","t-res-api/installation","t-res-api/usage"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":4,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":3,"sphinx.domains.rst":2,"sphinx.domains.std":2,"sphinx.ext.intersphinx":1,sphinx:56},filenames:["experiments/index.rst","getting-started/complete-tour.rst","getting-started/index.rst","getting-started/installation.rst","getting-started/resources.rst","index.rst","reference/geoparser/index.rst","reference/geoparser/linker.rst","reference/geoparser/pipeline.rst","reference/geoparser/ranker.rst","reference/geoparser/recogniser.rst","reference/index.rst","reference/utils/deezy_processing.rst","reference/utils/get_data.rst","reference/utils/index.rst","reference/utils/ner.rst","reference/utils/preprocess_data.rst","reference/utils/process_data.rst","reference/utils/process_wikipedia.rst","reference/utils/rel/entity_disambiguation.rst","reference/utils/rel/index.rst","reference/utils/rel/mulrel_ranker.rst","reference/utils/rel/utils.rst","reference/utils/rel/vocabulary.rst","reference/utils/rel_e2e.rst","reference/utils/rel_utils.rst","t-res-api/index.rst","t-res-api/installation.rst","t-res-api/usage.rst"],objects:{"geoparser.linking":{Linker:[7,0,1,""],RANDOM_SEED:[7,2,1,""]},"geoparser.linking.Linker":{by_distance:[7,1,1,""],load_resources:[7,1,1,""],most_popular:[7,1,1,""],run:[7,1,1,""],train_load_model:[7,1,1,""]},"geoparser.pipeline":{Pipeline:[8,0,1,""]},"geoparser.pipeline.Pipeline":{format_prediction:[8,1,1,""],run_candidate_selection:[8,1,1,""],run_disambiguation:[8,1,1,""],run_sentence:[8,1,1,""],run_sentence_recognition:[8,1,1,""],run_text:[8,1,1,""],run_text_recognition:[8,1,1,""]},"geoparser.ranking":{Ranker:[9,0,1,""]},"geoparser.ranking.Ranker":{check_if_contained:[9,1,1,""],damlev_dist:[9,1,1,""],deezy_on_the_fly:[9,1,1,""],find_candidates:[9,1,1,""],load_resources:[9,1,1,""],partial_match:[9,1,1,""],perfect_match:[9,1,1,""],run:[9,1,1,""],train:[9,1,1,""]},"geoparser.recogniser":{Recogniser:[10,0,1,""]},"geoparser.recogniser.Recogniser":{create_pipeline:[10,1,1,""],ner_predict:[10,1,1,""],train:[10,1,1,""]},"utils.REL.entity_disambiguation":{EntityDisambiguation:[19,0,1,""],RANDOM_SEED:[19,2,1,""]},"utils.REL.entity_disambiguation.EntityDisambiguation":{get_data_items:[19,1,1,""],normalize_scores:[19,1,1,""],predict:[19,1,1,""],prerank:[19,1,1,""],train:[19,1,1,""],train_LR:[19,1,1,""]},"utils.REL.mulrel_ranker":{MulRelRanker:[21,0,1,""],PreRank:[21,0,1,""]},"utils.REL.mulrel_ranker.MulRelRanker":{forward:[21,1,1,""],loss:[21,1,1,""],regularize:[21,1,1,""],training:[21,2,1,""]},"utils.REL.mulrel_ranker.PreRank":{forward:[21,1,1,""],training:[21,2,1,""]},"utils.REL.utils":{STOPWORDS:[22,2,1,""],flatten_list_of_lists:[22,3,1,""],is_important_word:[22,3,1,""],make_equal_len:[22,3,1,""]},"utils.REL.vocabulary":{Vocabulary:[23,0,1,""]},"utils.REL.vocabulary.Vocabulary":{add_to_vocab:[23,1,1,""],get_id:[23,1,1,""],normalize:[23,1,1,""],size:[23,1,1,""],unk_token:[23,2,1,""]},"utils.deezy_processing":{create_training_set:[12,3,1,""],generate_candidates:[12,3,1,""],obtain_matches:[12,3,1,""],train_deezy_model:[12,3,1,""]},"utils.get_data":{download_hipe_data:[13,3,1,""],download_lwm_data:[13,3,1,""]},"utils.ner":{aggregate_entities:[15,3,1,""],aggregate_mentions:[15,3,1,""],collect_named_entities:[15,3,1,""],fix_capitalization:[15,3,1,""],fix_hyphens:[15,3,1,""],fix_nested:[15,3,1,""],fix_startEntity:[15,3,1,""],training_tokenize_and_align_labels:[15,3,1,""]},"utils.preprocess_data":{aggregate_hipe_entities:[16,3,1,""],fine_to_coarse:[16,3,1,""],process_hipe_for_linking:[16,3,1,""],process_lwm_for_linking:[16,3,1,""],process_lwm_for_ner:[16,3,1,""],process_tsv:[16,3,1,""],reconstruct_sentences:[16,3,1,""],turn_wikipedia2wikidata:[16,3,1,""]},"utils.process_data":{align_gold:[17,3,1,""],eval_with_exception:[17,3,1,""],ner_and_process:[17,3,1,""],postprocess_predictions:[17,3,1,""],prepare_sents:[17,3,1,""],prepare_storing_links:[17,3,1,""],store_for_scorer:[17,3,1,""],update_with_linking:[17,3,1,""],update_with_skyline:[17,3,1,""]},"utils.process_wikipedia":{make_wikilinks_consistent:[18,3,1,""],make_wikipedia2wikidata_consisent:[18,3,1,""],title_to_id:[18,3,1,""]},"utils.rel_e2e":{get_rel_from_api:[24,3,1,""],match_ent:[24,3,1,""],match_wikipedia_to_wikidata:[24,3,1,""],postprocess_rel:[24,3,1,""],rel_end_to_end:[24,3,1,""],run_rel_experiments:[24,3,1,""],store_rel:[24,3,1,""]},"utils.rel_utils":{add_publication:[25,3,1,""],eval_with_exception:[25,3,1,""],get_db_emb:[25,3,1,""],prepare_initial_data:[25,3,1,""],prepare_rel_trainset:[25,3,1,""],rank_candidates:[25,3,1,""]},utils:{preprocess_data:[16,4,0,"-"]}},objnames:{"0":["py","class","Python class"],"1":["py","method","Python method"],"2":["py","attribute","Python attribute"],"3":["py","function","Python function"],"4":["py","module","Python module"]},objtypes:{"0":"py:class","1":"py:method","2":"py:attribute","3":"py:function","4":"py:module"},terms:{"0":[1,4,8,9,10,15,16,17,19,22,23,27,28],"00005":[1,10],"00021915406530791147":4,"0006031363088057901":8,"0015340784571553803":4,"0026298487836949377":4,"003935458480913026":9,"005478851632697786":4,"0075279261777561925":8,"007899999618530273":4,"00989999994635582":4,"010081087004163929":4,"01140":4,"013513513513513514":1,"014700000174343586":4,"022222222222222223":1,"03":[9,17],"03125":4,"03382000000000005":9,"03434":4,"039":1,"04":3,"042":1,"045":1,"049":1,"05":10,"05780":4,"06170":4,"06484443152079093":1,"07":21,"07407407407407407":9,"085":4,"1":[3,5,8,9,15,17,19,21,22,26],"10":[1,4,10,16],"100":12,"104":4,"10715509_7":4,"10732214":17,"10732214_1":17,"10813493_1":16,"114":4,"12":16,"1218":4,"1218_poole1860":4,"127":4,"13420000672340393":4,"141":1,"15":[1,9,16],"1595":[1,21],"1604":[1,21],"1666666865348816":9,"1790":[9,17],"1800":4,"1808999925851822":4,"1810":4,"184":28,"1880":1,"1900":1,"193":17,"19380":4,"199":17,"19th":4,"19thc":[1,4,8],"1e":21,"2":[3,5,17,22,26],"20":[3,7,18,19,20,21,22,23,28],"2017":[1,21],"2018":[1,21],"2020":[1,7,17,19,20,21,22,23],"206":17,"20francisco":18,"20languag":18,"20scienc":18,"2197":1,"2200":1,"24330":4,"2619":[1,21],"2629":[1,21],"27s_last_theorem":18,"28program":18,"29":18,"293":1,"295":1,"3":[3,4,5,7,16,17,19,21,22,26],"30440":4,"31":9,"313":1,"3157894736842105":1,"3257000148296356":4,"327":1,"33":4,"3333333333333333":9,"37":4,"38":[4,17],"3896239_29":4,"39":4,"396":1,"4":[4,5,10,17,19,22,26],"40":[4,17],"42":[7,19],"43rd":[1,7,19,20,21,22,23],"4457":4,"45":28,"455":1,"457":1,"5":[1,4,9,22],"50":[1,9],"50dff4e":0,"514":4,"522":1,"54":1,"566667":1,"56th":[1,21],"58949":4,"5e":10,"6":[4,22],"60":[1,9],"63800":4,"66":4,"7":[3,4,21,22],"70":12,"74":1,"75":[4,9],"783333":1,"79":17,"8":[1,10,16,22],"80":[1,27],"8000":28,"8262498_11":4,"83":4,"85":[1,9],"87":4,"881":1,"9":[3,4,22],"97":4,"9767696690773614":4,"99270":4,"999":1,"99975187":[10,17],"999826967716217":17,"boolean":4,"case":[1,4,15,17,25],"class":[1,4,5,7,8,9,10,16,19,21,23],"default":[1,4,7,8,9,10,12,17,19,22,23,25,27,28],"do":[1,3,4,22],"export":[3,27],"final":[1,4,8],"float":[4,7,8,9,12,17,19,22],"function":[0,7,8,10,12,15,16,17,18,19,21,22,24,25],"import":[1,4,22,27],"int":[8,12,15,17,22,23,24],"long":[1,21],"new":[1,4,8,12,15,16,18,22,25,27],"null":3,"public":[1,4,7,8,16,17,25],"return":[0,1,7,8,9,10,12,13,15,16,17,18,19,21,22,23,24,25],"static":23,"true":[1,4,7,8,9,10,12,15,22,27],"try":[1,8],"while":22,A:[1,4,7,8,9,10,12,15,16,17,19,22,23,24,25],And:3,As:1,At:17,By:[1,4,8],For:[1,4,7,8,9,10,15,17],If:[1,3,4,7,8,9,10,12,15,16,17,18,25,28],In:[1,3,4],It:[1,7,8,9,10,12,15,19,25],On:4,The:[2,4,5,7,8,9,10,12,13,15,16,17,18,20,21,22,23,24,25,26,27,28],Then:3,There:[0,15],These:4,To:[0,1,3,4,27],_:[16,17],_gold_posit:17,_gold_standard:17,_imag:27,_ner_predict:17,_ner_skylin:17,_new:[1,9],_pred_ment:17,_router:27,_test:1,abl:1,about:[1,4,9,17,22,25],abov:[3,4,7,17,22],absolut:4,access:[4,8,26],accord:[1,17,27,28],accordingli:[1,9],account:[1,16],accur:1,achiev:17,acm:[1,7,19,20,21,22,23],across:[3,9,22],activ:3,actual:[4,16],ad:[0,1,8,17,23,25],adapt:[1,4,7,10,15,18,19,20,21,22,26],add:[3,4,21,23,25,27],add_publ:25,add_to_vocab:23,addit:[1,10],addition:9,address:15,affect:4,after:[4,8,16,18,22],afterward:22,again:22,against:22,aggreg:[15,16],aggregate_ent:15,aggregate_hipe_ent:16,aggregate_ment:15,agnost:1,algorithm:1,align:[10,15,17],align_gold:17,all:[0,1,3,4,7,8,15,16,17,22],all_cand:1,all_test:17,all_toponym:1,allow:[1,4,26],almost:22,alon:22,along:[8,16,22,25],alreadi:[1,4,7,8,9,10,22],already_collect:9,already_collected_cand:9,also:[1,3,4,5,8,9,10,15,17,22,26],alston:1,altern:[1,4,9],although:22,altnam:9,alwai:[1,4,22,27],am:[1,22],among:[17,22],amongst:22,amoungst:22,amount:[9,22],an:[4,5,7,8,9,10,12,15,16,17,19,20,21,22,23,25],analog:17,anchor:[1,4],ancient_egypt:16,ani:[1,8,10,17,18,22,25],ann:4,annot:[4,16,17,24],annotated_tsv:16,annual:[1,21],anoth:[17,22],anyhow:22,anyon:22,anyth:22,anywai:22,anywher:22,api:[5,24],api_usag:28,app:[4,26,27,28],app_nam:27,app_templ:[26,27],appear:[1,4],appl:[9,22],appli:[1,4,7,19],applic:[9,28],approach:[0,1,4,7,9,21,24,25],appropri:[9,15],apt:3,ar:[0,1,3,4,5,8,9,10,12,15,16,17,19,20,21,22,25,26,27],architectur:19,arg:27,argument:[1,10,12],arjen:[1,7,19,20,21,22,23],around:22,arrai:[4,22,25],arriv:4,articl:[4,16,17,24,25],article_id:[4,16,17],articles_test:17,ashton1860:24,ashton:[1,4,15],assess:17,assign:[4,7,9,15,16,17],associ:[1,7,8,16,21,23],assum:[1,4,16],ast:17,asterisk:4,attent:[1,21],attribut:[9,10],author:[7,19,20,21,22,23],automat:[4,28],avail:[16,21,27,28],axi:1,b:[4,15,16,17],back:[1,4,22],bad:1,balanc:27,balog:[1,7,19,20,21,22,23],banana:9,barcelona:9,barnett:28,base:[1,2,5,7,9,10,12,15,16,17,19,21,23,24],base_model:[1,10],bash:[3,27],bashrc:3,basi:[1,17],basic:3,batch:[1,21],batch_siz:[1,10],batchencod:15,becam:22,becaus:[1,15,17,19,22],becom:22,been:[1,3,4,5,8,9,17,20,22],befor:[1,4,22,25,27],beforehand:[1,22],begin:15,behaviour:27,behind:[22,27],being:[1,22],belong:[4,16],below:[1,4,7,8,9,12,22],bert:[1,4,10,17],bert_1760_1900:1,besid:22,best:[1,22],better:1,between:[1,4,9,12,19,21,22],beyond:22,billion:4,bin:3,bio:[4,16],bl:13,blb_lwm:1,bologna:9,booktitl:[7,19,20,21,22,23],bool:[7,8,9,10,17,18,21,22,23],both:[1,4,15,19,22,25],bottom:22,british:4,build:[3,4,5,17,26],built:[4,27],by_dist:7,bydist:7,byrom:4,c:[7,19,20,21,22,23],cach:27,cadiz:4,calcul:[7,9],call:[1,4,9,12,15,22,27],can:[1,3,4,8,15,22,25,26,27],cand:1,candid:[4,5,7,8,9,12,17,19,21,25],cannot:22,cant:22,capit:15,captur:1,care:[0,1],cd:3,centenni:4,centuri:[1,4],cerimoni:4,certain:[1,7],chang:[1,3,4,15],charact:[4,10,15,16,17,18,22,24],characterist:1,characters_v001:1,check:[9,22],check_if_contain:9,choos:17,chosen:9,christo:4,church:4,citi:[1,4,8,16,18],classifi:12,clean:1,clef:17,clone:[0,3],closest:7,cloth:4,co:22,coach:4,coars:16,code:[0,1,4,5,16,21,23],codebas:3,coeld:4,collect:[9,15],collect_named_ent:15,colleg:4,collegi:4,colosseum:16,column:[1,4,9,16],com:[3,12,17,18],combin:[1,5,15],come:[1,4,8],comma:10,command:[0,3,27],commit:[0,2,5],common:[1,4,5],complain:4,complet:[2,5,16],complex:1,compon:[1,5],compos:[5,26],comput:[1,9,10,19,21],con:22,conf_md:[1,8],confer:[1,7,19,20,21,22,23],confid:[1,7,8,10,19],config:[21,26,27],config_nam:26,configur:[5,26],conld:4,conll:[17,24],conn:[1,4,7],connect:[1,4,7,15,25],consecut:16,consid:[1,12,15],consider:1,consist:[3,4,12,15,16,17,18,26],consolid:15,contain:[1,4,5,7,8,9,10,12,15,16,17,22,24,25,26],container_nam:27,content:[4,28],context:[1,7,8,19,21],contigu:16,continu:4,convert:[16,18,23,25],coordin:[1,4,5,8],copper:4,copyright:[7,19,20,21,22,23],core:[9,16,17,25],corpu:4,correct:[15,28],correspond:[4,5,7,8,9,12,15,16,17,18,19,24,25],corrrect:19,couid:4,could:[1,4,18,22],couldnt:22,council:4,count:[4,25],counti:4,creat:[0,1,3,4,7,9,10,12,16,17,28],create_pipelin:10,create_training_set:12,credit:[7,10,15,18,19,21,22,23],cross:[1,8],cross_cand_scor:[1,8],cry:22,csv:1,ctx_layer:21,cumbria:1,curl:[3,28],current:[15,16,17],cursor:[1,4,7,25],customis:1,d:[27,28],damerau:9,damlev:9,damlev_dist:9,dannot:17,dash:10,data:[1,2,5,7,9,12,16,17,18,19,25],data_path:[1,7],data_sci:18,databas:[1,4,7,25],datafram:[1,16,17,25],dataset:[0,8,9,10,12,13,15,19,24,25],date:3,db:[1,4,7,18],db_emb:19,db_embed:[1,7],de:[1,7,19,20,21,22,23],decad:[4,16],decai:1,decim:7,decod:18,deep:[1,9,19,21],deezy_on_the_fli:9,deezy_paramet:[1,9,12],deezy_process:[5,11,14],deezymatch:[2,5,9,12],deezymatch_on_the_fli:[1,9],default_publnam:[1,7],default_publwqid:[1,7],defin:[3,12],degre:9,deleg:9,delici:9,depend:[1,3,9],deploi:[5,28],deploy:[5,26],depth:1,dercksen:[1,7,19,20,21,22,23],deriv:4,describ:[1,4,5,7,22],descript:[4,15],design:5,detail:[1,7,9,15,22],detect:[1,4,15,16,17],determin:7,dev:[3,4],dev_json:19,develop:[1,4,7,19,20,21,22,23],deviat:16,devic:21,df:[1,4,17,25],dict:[1,7,8,9,10,12,15,16,17,24,25],dict_ment:7,dictionari:[1,4,7,8,9,10,12,15,16,17,24,25,27],did:22,didnot:4,differ:[0,1,4,7,9,15,25],digit:23,digit_0:23,digitis:4,dipend:3,directli:[1,3,4],directori:[0,1,2,5,9,12,16,27],disamb_output:1,disambigu:[2,5,7,8,19,21,25],discard:12,display_result:0,distanc:[1,7,9],divid:4,dm_cand:[1,9],dm_model:[1,9],dm_output:[1,9],dm_path:[1,9,12],dmentionsgold:17,dmentionspr:17,dmetadata:17,dmtoken:16,dname:19,do_test:[1,7,9,10],doc:[9,28],docker:[5,26],dockerfil:[26,27],docstr:1,document:[0,1,4,5,8,9,16,28],document_dataset:8,document_id:16,doe:[0,1,12,19],don:4,done:[1,22],dont:22,down:22,download:[0,1,4,13],download_hipe_data:13,download_lwm_data:13,dpred:17,drel:24,dresult:17,dsentenc:[17,24],dsky:17,dsplit:25,dtoken:16,dtrue:17,due:22,dukiafield:4,dukinfield:4,dummi:4,dure:[8,15,17,22,23],durham:1,e:[1,3,4,7,8,9,12,15,16,17,18,19,24,25,27],e_typ:15,each:[1,4,5,8,9,10,15,16,17,19,22,24,25],easier:1,echo:3,ed:[4,19],ed_model:1,ed_scor:[1,8],edit:27,effici:1,eg:22,eight:22,either:[1,4,15,22,27],element:[5,15,16,17,24,25],eleven:22,eliza:4,elizabeth:28,els:[4,22],elsewher:22,emb:4,embed:[1,2,5,7,12,21,23,25],embeddingbag:22,embeddings_databas:[1,4,7],embtyp:25,empir:[1,21],empti:[1,4,7,8,9,15,16,17,22,25],en:[1,8,16,18],enabl:[17,27],encod:[15,18],end:[0,4,5,8,10,15,16,17,22,24],end_char:[15,17],end_offset:[15,17],end_po:[1,8],end_to_end_ev:17,endpoint:[27,28],england:[1,4],english:[4,12],english_label:4,english_word:12,enough:22,ensur:16,ent_scor:21,entir:8,entiti:[2,5,7,8,9,10,12,15,16,17,18,19,20,21,22,23,24,25],entity2class:1,entity_disambigu:[7,14,20],entity_embed:4,entity_id:21,entity_link:[15,17],entity_mask:21,entity_typ:[4,15],entitydisambigu:[7,19],entityt:7,entri:18,env:3,environ:[3,27],environemnt:3,epoch:1,equal:22,equat:21,equival:16,error:[15,17,25],errror:1,essenti:3,establish:7,etc:22,etiti:9,eugen:[1,21],eval:3,eval_stat:0,eval_with_except:[17,25],evalu:[4,5,10,17],even:[1,22],ever:22,everi:[4,22],everyon:22,everyth:22,everywher:22,exact:[1,9],exampl:[1,4,7,8,9,10,15,16,17,18,22,26,27,28],except:[1,4,22],exclud:[8,25],execut:[4,7,9,10],exist:[7,9,10,12,16,18,19],expand:4,expect:[1,4,16],experi:[1,3,4,5,7,16,17,24,25,27],explicitli:3,expos:[27,28],express:[1,17],extern:5,extract:[4,8,9,17,25],f:[4,21,27],faegheh:[1,7,19,20,21,22,23],faiss:[1,9],fals:[1,4,7,8,9,10,18,19,22,23],fastapi:26,fastest:1,featur:1,fermat:18,fetchon:4,few:[1,3,22],field:[1,4,15,25],fifi:22,fifteen:22,figur:21,file:[1,3,4,12,16,17,24,25,26,27],filenam:1,filepath:16,fill:[17,22],fill_in:22,filter:[1,9,12],find:[1,4,8,9,22,24],find_candid:[1,8,9],fine:[1,10,16],fine_to_coars:16,fire:22,first:[1,3,4,8,9,12,15,17,22],fit:26,five:[4,17,22],fix:15,fix_capit:15,fix_hyphen:15,fix_nest:15,fix_startent:15,flag:[7,9,23],flatten:22,flatten_list_of_list:22,fly:9,focus:4,folder:[0,1,4,12,17],follow:[0,1,3,4,8,9,15,16,17,18,26,27,28],form:[15,25],format:[0,1,4,8,15,16,17,19,24,25],format_predict:8,former:22,formerli:22,forti:22,forward:21,found:[1,4,8,9,18,22,23,24,25],four:[1,4,22],fragment:18,frame:[16,17,25],frequenc:[4,9],from:[0,4,7,8,9,10,12,13,15,16,17,18,19,20,21,22,23,24,25,27,28],front:22,full:[1,8,17,22,24],further:[9,22],futur:8,fuzz:12,fuzz_ratio_threshold:12,fuzzi:[1,4,9],fuzzywuzzi:1,g:[1,4,7,8,9,15,16,17,18,21,24,25],ganea2017deep:21,ganea:[1,21],gazett:[1,16,24],gazetteer_id:[16,24],gener:[4,9,12,16,25,28],generate_candid:12,geograph:[1,4,5,7,8],geopars:[1,4,5,11,17,25,27],get:[1,5,19,22,23,28],get_data:[5,11,14],get_data_item:19,get_db_emb:25,get_id:23,get_rel_from_api:24,get_result:0,giant:[1,7,19,20,21,22,23],girl:4,git:3,github:[3,4,7,12,17,18,19,20,21,22,23],give:[4,22],given:[5,7,8,9,10,12,16,17,18,19,21,23,25],global:[1,3],go:[0,22],gold:[1,8,15,17,21,24,25],gold_posit:17,gold_token:[17,24],grain:16,ground:[4,19],group:15,guadaloup:9,guarante:3,guid:27,h:28,ha:[1,4,5,8,9,17,22],had:[7,22],hall:4,handl:[9,10],hardli:4,harvei:28,hasibi:[1,7,19,20,21,22,23],hasnt:22,have:[0,1,3,4,17,19,20,22],haven:[15,17],he:22,head:4,header:27,help:16,henc:22,her:22,here:22,hereaft:22,herebi:22,herein:22,hereupon:22,herself:22,hi:22,hide:4,high:[4,19],higher:1,him:22,himself:22,hipe:[0,13,16,17,24],hipe_path:[13,16],hipe_scorer_results_path:17,histor:[4,10],hmd:1,hofmann:[1,21],home:[3,9,27],hook:3,hoook:[2,5],host:27,host_url:27,hour:1,hous:4,how:[1,2,4,5,8,22,28],how_split:24,howev:[1,22],http:[3,12,16,17,18,27,28],hub:[1,4,10],huggingfac:[1,4,10,15],hulst:[1,7,19,20,21,22,23],human:[1,8],hundr:22,hyphen:15,i0001_9:17,i0004_1:9,i:[1,4,8,9,10,12,15,16,17,22],id:[1,4,5,7,8,9,12,15,16,17,18,23,24,25],ideal:1,ident:1,identifi:[1,4,5,8,9,15,16,18,24,25],identified_toponym:1,ie:22,ignor:[1,19],imag:27,implement:[1,9,21,25],impresso:17,imprison:4,improv:[1,4,21],in_cas:[17,25],inc:22,includ:[8,9,10,16,20,22],incorpor:17,incorrect:15,incorrectli:15,inde:22,index:[1,4,5,8,15],index_enwiki:4,india:16,indic:[1,4,7,9,16,23],individu:1,infer:1,inform:[1,4,7,9,15,16,17,19,20,21,22,23,24,25],init:3,initi:[7,10,17,25],initialis:[1,7,8,9,10],inlink:1,inner:[4,17,24],inproceed:[7,19,20,21,22,23],input:[1,5,8,10,12,15,22,24,25],input_dfm:[1,12],insid:[12,16],inspector:1,instal:[2,5,26,27],instanc:[3,10,15],instanti:[4,7,8,27],instead:[15,17,26],instruct:[0,3,4,8],interact:28,interest:[4,22],intern:[1,7,19,20,21,22,23],invalid_loc:16,ipynb:[1,28],ipython:3,is_important_word:22,island:15,issu:[10,15],item:1,iter:15,its:[1,4,8,9,12,16,17,19,21,22,23],itself:[1,9,22],ivan:[1,21],jcklie:18,johann:[1,7,19,20,21,22,23],join:[15,16],joint:[1,21],json:[1,10,17,25,28],jsth:4,jupyt:3,just:[1,22],keep:[15,17,22],kei:[1,4,7,8,9,10,15,16,17,24,25],kernel:3,kernel_nam:3,keyword:[8,12],kingdom:[4,7],know:1,knowledg:[1,7,9,19,24],known:22,knutsford:4,koen:[1,7,19,20,21,22,23],krisztian:[1,7,19,20,21,22,23],l:[1,16],label2id:15,label:[1,4,8,10,15,16,17,27],label_encoding_dict:15,lamb:21,lambda:1,languag:[1,8,12,21],larg:1,larger:8,last:[8,18,22],latent:[1,21],later:[17,22],latest:[4,27],latex:0,latitud:[1,4,8],latlon:[1,8],latter:22,latterli:22,le2018improv:21,le:[1,21],learn:[1,4,19],learning_r:[1,10],least:[1,4,22],leav:4,left:[1,4],left_out:4,legibl:[1,8],length:[1,4,22],lentiti:[15,16],lerwick:4,less:[1,22],let:1,level:[1,8,16],levenshtein:9,libbz2:3,libffi:3,liblzma:3,libncursesw5:3,librari:[1,3,4],libreadlin:3,libsqlite3:3,libssl:3,libxml2:3,libxmlsec1:3,liddl:1,like:[1,4,8,17,19],likewis:4,line:[0,1,4],linguist:[1,21],link:[1,4,5,6,8,9,11,15,16,17,18,21,24,25],link_predict:17,linker:[2,4,5,6,8,11,19,20,21,22,23,27],linking_df_split:[1,4],linking_resourc:[1,7],linux:3,list:[1,4,8,9,10,12,15,16,17,19,22,24,25],list_of_list:22,liter:[7,9,15,25],literal_ev:17,littl:4,live:[1,3,10],livingwithmachin:[1,8],llvm:3,load:[4,7,9,10,27],load_from_hub:[1,8,10],load_from_path:1,load_resourc:[1,7,9],load_to_hub:1,load_use_ner_model:1,loadbalanc:27,loc:[1,4,15,16,17],local:[1,4,10,21,26],locat:[1,4,7,12,16],lodg:4,london:[1,4,8,9,10,28],longer:1,longitud:[1,4,8],look:[1,17],loss:21,lot:1,low:19,lower:[4,18,23],lowercas:[9,18,23,25],lr:19,ltd:22,lwm:[0,1,4,7,13,16,17],lyne:[1,4,15],m:[1,7,19,20,21,22,23],machin:3,made:22,mai:[1,15,22],main:[1,5],maintain:[4,15],make:[1,3,4,18,22],make_equal_len:22,make_wikilinks_consist:18,make_wikipedia2wikidata_consis:18,make_wikipedia2wikidata_consist:18,manag:[3,7],manate:18,mancheft:1,manchest:[4,28],mani:22,manner:1,manual:[15,17],map:[4,8,9,12,15,16,17,18,24,25],mapper:18,mark:4,mask:22,match:[1,4,8,9,12,16,24],match_ent:24,match_scor:9,match_wikipedia_to_wikidata:24,max:19,max_len:[1,9],max_norm:21,maximum:[1,17],me:22,mean:[4,16],meanwhil:[4,22],measur:[1,4],meet:[1,21],memori:4,ment:1,mention:[4,5,7,8,9,12,15,17,19,21,25],mention_already_collect:9,mention_candid:9,mention_end:4,mention_po:4,mention_start:4,mentions_to_wikidata:[1,9,12,25],mentions_to_wikidata_norm:1,metadata:[16,17],method:[1,4,7,8,9,10,17,19,21,23],meto_typ:16,metric:[1,9,10],michael:[7,19,20,21,22,23],microtoponym:[1,8],middlewar:27,might:[4,22],mill:22,millgat:4,min_len:[1,9],mine:22,minim:21,minimum:1,mock:1,mode:[1,7,9,10],model:[4,7,8,9,10,12,15,16,17,19,21,25],model_path:[1,7,10],model_path_lr:19,model_state_dict:1,modern:1,modifi:18,modul:[5,11],monteveido:4,month:4,more:[1,4,7,9,17,22,23,28],moreov:22,morn:4,most:[1,4,7,19,22],most_popular:7,mostli:22,mostpopular:[4,7,8],move:22,much:22,mulrel:21,mulrel_rank:[14,20],mulrelrank:[19,21],multi:[16,21],multipl:[5,15,26],multipli:21,must:22,my:22,mylink:[1,7,8,25],myner:[1,8,17],myrank:[1,7,8,9,12,25],myself:22,n:[3,10,12],name:[1,3,4,5,8,9,10,15,16,17,21,22,25,27],namedtupl:15,natur:[1,8,21],ndarrai:25,ne_typ:16,nearest:12,necessari:[1,4,7,21],need:[0,1,3,4,7,9,16,25,26],neg:[1,4,12],neighbor:12,neighbour:12,neither:22,nel:21,ner:[4,5,8,10,11,14,16,17],ner_and_process:17,ner_fine_dev:[1,4],ner_fine_test:4,ner_fine_train:[1,4],ner_label:[15,17],ner_output:1,ner_predict:[10,17],ner_scor:[1,8,15,17],ner_tag:[4,15,16],nest:[8,15],network:[9,21],neural:[1,9,21],never:[4,22],nevertheless:22,new_york_c:18,news_dataset:[1,4],news_path:13,newspap:[4,10],next:[1,4,8,22],nfi:3,ngram:[1,8],nil:[1,7,24],nine:22,nineteenth:1,nlp_df:1,nn:22,nobodi:22,nois:[4,9],none:[1,4,7,8,9,10,12,13,16,17,18,19,21,22,23,24,25],noon:22,nor:22,norm:1,normal:[9,19,23],normalis:[4,8,23],normalize_scor:19,north:4,note:[1,4,7,8,9],notebook:[1,3,28],noth:22,notic:[1,7,19,20,21,22,23],novemb:4,now:[3,4,8,17,22,28],nowher:22,np:25,npy:[1,4],num_candid:[1,9],num_train_epoch:[1,10],number:[1,4,9,16,22,23],numpi:25,o:[4,10,15,16,17],object:[1,4,7,8,9,10,12,15,17,23,24,25],obtain:[1,4,5,10,12],obtain_match:12,occur:[15,17],occurr:16,oclock:4,ocr:[1,4,10,12,16],ocr_quality_mean:[16,17],ocr_quality_sd:[16,17],ocr_threshold:[1,9],octavian:[1,21],off:22,offici:27,offset:[15,22,24],often:[1,22],old:4,onc:[1,22],one:[0,1,4,8,15,16,17,22],onli:[4,15,17,22],onto:22,open:[4,25],oper:18,option:[4,7,8,9,10,12,16,17,18,22,23,25,27],orang:9,order:[1,3,4,7,16,17],org:[3,16,18],org_dev_dataset:19,org_train_dataset:19,origin:[4,7,8,9,15,16,17,21,23,25],origin_wqid:7,originalsplit:[1,4,7,24],other:[1,4,22],otherwis:[4,9,10,17,18,22],our:[0,1,3,4,16,22],ourselv:22,out:[1,4,22],outermost:8,output:[4,7,8,12,17,24],over:[15,19,22],overlap:[1,9,17],overrid:1,overview:18,overwrit:[1,7,10],overwrite_dataset:[1,9],overwrite_train:[1,7,9,10,12],own:[4,17,22,26],p:[1,7,19,20,21,22,23,27],p_e_m:21,packag:3,pad:22,page:[0,1,4,5,16,18,21],page_titl:18,pair:[4,9,12,17],panda:[4,9,16,17,25],pandarallel:9,paper:[0,1,21],paraguai:9,paramet:[1,7,8,9,10,12,13,15,16,17,18,21,22,23,24,25],pari:8,pars:[10,25],part:[15,16,17,18,22],partial:[1,9],partial_match:9,partialmatch:9,particular:4,pass:[1,8,9,12,16,21,25],path:[1,3,4,7,9,10,12,13,16,17,18,24,27],path_to_db:18,pathlib:1,pathprefix:27,pd:[1,4,9,17],per:[1,4,16,17,19,21,22,24],percent:18,perfect:9,perfect_match:9,perfectli:9,perfectmatch:[4,8,9],perform:[1,4,5,7,8,9,10,15,16,17,18,19,24,25],perhap:22,period:4,permiss:[7,19,20,21,22,23],petr:17,phong:[1,21],phrase:15,pipe:10,pipelin:[2,4,6,10,11,17,26,27],place:[1,4,5,7,8,16,17,28],place_wqid:[1,4,8,17,28],plain:[4,24],plan:1,pleas:[4,22],plu:4,po:[1,8],poetri:[2,5,27],point:[8,17,25],polic:1,popular:[1,7],port:[27,28],posit:[1,4,8,10,12,15,16,17,24],possibl:[19,25],post:[4,8],postprocess:[8,17],postprocess_output:8,postprocess_predict:17,postprocess_rel:24,potenti:8,pp:1,pre:[1,2,4,5,10],preappend:25,preced:4,pred:[15,17],pred_ent:24,predict:[1,4,7,8,9,10,15,17,19,24],prefer:4,prefix:[15,16,25,27],prepar:[1,5,17,25],prepare_data:[0,25],prepare_initial_data:25,prepare_rel_trainset:25,prepare_s:17,prepare_storing_link:17,preposit:15,preprocess:27,preprocess_data:[5,11,14],prerank:[19,21],present:[0,18,19],pretrainedtoken:15,pretrainedtokenizerfast:15,prev_ann:24,previou:[1,8,15,16,24,28],print:[1,8,9,10,16,22],prior:[1,8],prior_cand_scor:[1,8],probabl:1,problem:[5,15],proceed:[1,7,19,20,21,22,23,27],process:[1,4,7,8,9,10,15,16,17,21],process_data:[5,11,14],process_hipe_for_link:16,process_lwm_for_link:16,process_lwm_for_n:16,process_tsv:16,process_wikipedia:[5,11,14],processed_data:[8,17],produc:[17,21],programming_languag:18,progress_appli:1,project:[2,5],properli:3,propos:1,provid:[0,1,4,5,7,8,9,10,12,15,17,25,26,27],proxi:27,publication_cod:[16,17],publication_titl:[16,17],publish:[7,19,20,21,22,23],publnam:25,publwqid:25,put:22,py:[0,3,25,26,27],pyenv:[2,5],pyenv_root:3,pytest:3,python3:3,python:[0,3,4,18,28],python_:18,q10285:16,q1075483:1,q1137286:1,q11768:16,q145:7,q1470791:4,q17003433:1,q17012:9,q179815:1,q180673:4,q18125:28,q1976179:4,q201970:8,q23082:1,q23103:8,q23183:[1,4],q2477346:9,q2560190:1,q30:17,q3153836:9,q335322:17,q458393:1,q49229:1,q5059107:4,q5059144:4,q5059153:4,q5059162:4,q5059178:4,q515:[1,4],q5316459:1,q5316477:1,q55448990:[1,4],q60:16,q752266:1,q8023421:[1,4],q84:[1,4,7,8,9],q:21,qid:[1,4,24],qualiti:16,quantitav:1,quantiti:4,quarter:4,queri:[1,9,25,28],question:[1,4],quit:1,quot:18,r:4,radboud:[7,19,20,21,22,23],random_se:[7,19],randomli:12,rang:9,rank:[1,5,6,7,8,11,17,21,25],rank_candid:25,ranker:[2,5,6,7,8,11,12,25,27],ranking_metr:[1,9],rate:1,rather:22,ratio:[1,12,19],re:[1,2,4,9,22],read:[4,16],read_csv:4,read_pickl:1,readi:1,readm:1,realli:1,recal:19,recognis:[2,5,6,8,11,17,27],recognit:[1,2,5,8,10,15,16,17],reconstruct:[15,16],reconstruct_sent:16,redund:8,ref:27,refer:[1,4,5,7,9,19,20,21,22,23],regard:15,regardless:1,regular:[1,21],rel:[1,4,5,7,9,11,14,24,25],rel_db:[1,4,7],rel_e2:[5,11,14],rel_end2end_path:24,rel_end_to_end:24,rel_json:25,rel_param:[1,7,8,25],rel_pr:24,rel_util:[5,11,14],relabel:17,relat:[1,8,9,10,21],reldisamb:[7,25],relev:[7,8,9],remain:[8,9],remot:26,remov:[9,18],render:4,replac:[10,15,18,23],repo:3,report:0,repositori:[4,7,13,19,20,21,22,23,26,27],repres:[8,9,10,15,17,23,25],represent:17,reproduc:0,requir:[0,1,4,7,8,9,12,17,19,24,25,27],res_:27,res_deezy_reldisamb:[27,28],rescal:19,research:[1,4,7,19,20,21,22,23],reserv:15,reset_embed:19,resolut:[4,16],resolv:[1,4,9,19,28],resourc:[2,5,7,8,9,27],resources_path:[1,7,8,9],respect:[1,10,17],respons:[19,21],restart:[3,27],result:[0,1,4,7,8,9,12,15,17,24],retoken:24,retriev:[7,12,17,19,20,21,22,23,24,25],reus:4,revers:27,right:[1,22],root:27,round:7,router:27,row:[1,4,9,16,17],rule:[23,27],run:[1,3,4,5,7,8,9,10,24,27],run_candidate_select:[1,8],run_disambigu:[1,8],run_rel_experi:24,run_sent:[1,8],run_sentence_recognit:8,run_text:[1,8],run_text_recognit:[1,8],runner:9,rwop:27,s:[1,4,7,8,9,15,22,27,28],s_last_theorem:18,said:1,salop:8,same:[1,4,15,16,17,22],samsung:4,san:18,san_francisco:18,santo:4,save:[1,4,10,12,24],scenario:0,scenario_nam:17,scheme:16,school:4,scienc:18,score:[1,4,7,8,9,10,15,16,17,19,21],score_combin:21,scorer:[0,17,24],script:[0,3,4,16,20,23,25],search:5,seatgeek:12,second:[8,9,12,17],section:[1,4,27,28],see:[1,4,7,8,9,15,17,19,20,21,22,23,28],seem:22,select:[1,3,4,5,7,8,9,12],selection_threshold:[1,9],self:[1,24],sens:1,sent:24,sent_idx:[1,8],sent_po:4,sentenc:[1,4,8,9,10,15,16,17,24,28],sentence_id:[16,17],sentence_po:[4,17],sentence_pr:17,sentence_ski:17,sentence_text:4,sentence_tru:17,separ:24,sequenc:15,seri:[4,7,9,17,19,20,21,22,23],seriou:22,servant:4,server:[27,28],servic:27,session:[3,4],set:[1,2,5,7,8,9,10,12,15,16,17,19,21,24,25,27],setup:8,sever:[4,22],share:4,she:[4,22],sheffield:9,shefrield:9,shell:3,ship:4,should:[0,1,4,7,15,19,22,23,27,28],shoulder:[1,7,19,20,21,22,23],show:[4,5,8,9,22,28],side:22,sigir:[1,7,19,20,21,22,23],sign:4,significantli:1,sim:12,similar:[1,8,9,12,17],simpl:12,simultan:27,sinc:[1,22],sincer:22,singl:[4,8,15,28],six:[16,22],sixti:22,size:[1,23],skip:[1,4,7,9,10],sky:17,skylin:17,slow:1,sn83030483:[9,17],snd:25,so:[1,22],solut:15,some:[4,15,22],somehow:22,someon:22,someth:22,sometim:22,somewher:22,soon:4,sourc:[3,16],southampton:4,space:[15,16,18],special:18,specif:[1,15,16,19,24],specifi:[1,4,7,8,9,10,12,23,25],split:[1,4,7,8,9,15,17,24,25],sqlite3:[1,4,7,25],ssl:3,st:22,stand:[1,7,19,20,21,22,23],standalon:27,standard:[8,16,17,24],start:[1,4,5,8,10,15,16,17,24],start_char:[15,17],start_offset:[15,17],state:17,step:[0,4,8,17,19,27,28],still:[1,22],stole:4,stopword:22,store:[1,4,9,10,12,17,23,24,25,27],store_for_scor:17,store_rel:24,str2pars:[17,25],str:[1,7,8,9,10,12,13,15,16,17,18,22,23,24,25],strategi:1,street:4,string:[4,8,9,12,15,16,17,24,25],string_match_scor:[1,8],stripprefix:27,structur:[1,2,5,7,16,27],strvar_paramet:[1,9,12],style:3,sub:9,subsampl:1,subset:[1,8],subtract:9,succe:17,success:[17,25],sudo:[3,27],suffix:[1,17],suggest:3,sum:19,summari:[2,5],summer:8,sundai:4,suppos:1,sure:[1,3,4,28],surfac:15,swagger:28,symbol:18,syn1neg:[1,4],system:[2,5,9,22],t:[1,2,4,9,15,17],tabl:[0,4],tackl:5,tag:[1,8,15,16,17],take:[0,1,8,9,10,15,16,22,25],taken:[1,19,20,21,22,23],target:[1,8],task:[1,4,7,8,10,19],tell:3,templat:[26,27],ten:[4,22],terceira:15,term:[1,21],test:[1,3,4,7,9,10,17,25,27],test_dataset:[1,10],test_df:17,text:[1,4,5,7,8,9,10,15,16,17,18,19,24,28],than:[1,19,22],thefuzz:12,thei:[1,4,9,17,18,22],them:[0,1,3,4,5,8,9,12,15,16,19,22,27],themselv:22,thenc:22,thereaft:22,therebi:22,therefor:[1,4,22],therein:22,thereupon:22,thi:[0,1,4,5,7,8,9,10,12,15,16,17,18,19,20,21,22,23,24,25,26,27],thick:22,thin:22,third:22,thoma:[1,21,28],those:[1,9,15,22],though:22,three:[1,4,5,8,15,17,22,24],threshold:[1,9,12],through:[4,8,9,17,22],throughout:22,thru:22,thu:22,time:[1,4,8,19],titl:[7,16,18,19,20,21,22,23,24],title_to_id:18,titov:[1,21],tk:3,to_right:22,todo:1,togeth:22,tok_mask:21,token:[4,10,12,15,16,17,23,24],token_id:21,token_offset:21,tokenis:17,tokenization_util:15,tokenization_utils_bas:15,tokenization_utils_fast:15,tolist:4,too:22,top:[1,4,12,16,22],top_threshold:[1,9],toponym:[1,2,8,10,16,28],toponym_resolut:[0,3,28],topres19th:4,torch:22,tour:[2,5],toward:[4,22],traefik:27,train:[2,5,7,9,10,12,15,16,19,21,25],train_dataset:[1,10],train_deezy_model:12,train_json:19,train_load_model:[1,7],train_lr:19,train_use_deezy_model_1:1,train_use_deezy_model_2:1,train_use_deezy_model_3:1,train_use_ner_model:1,trainer:10,training_arg:[1,10],training_split:[1,7],training_tokenize_and_align_label:15,transform:[1,10,15],trickier:1,trigger:19,true_po:21,truncat:22,truth:19,tsv:[1,4,16,17,24],tsv_topres_path:16,tune:[1,10],tupl:[7,8,9,12,15,16,17,22,24],turn_wikipedia2wikidata:16,tutori:[4,10],twelv:22,twenti:22,two:[1,4,8,9,12,15,16,17,22],txt:[1,12],type:[1,4,7,8,9,10,12,15,16,17,18,19,22,23,24,25,28],ubuntu:3,uk:4,un:22,uncas:10,under:[1,4,15,22],underscor:[18,24],undertak:7,unescap:18,union:[12,15],uniqu:[1,16],unit:[4,7],unitec:17,unk:[4,23],unk_token:23,unknown:23,unless:[1,17],unprocess:17,unquot:18,unsupervis:[1,4,7],until:22,unzip:13,up:[1,3,4,8,12,22,27],updat:[2,5,9,16,17,27],update_with_link:17,update_with_skylin:17,updated_ent:16,upon:22,url:[16,18],us:[0,2,4,5,7,8,9,10,12,15,16,17,18,19,21,22,23,24,25,26,27],user:[1,3,4,26],user_config:19,usual:3,util:[3,4,5,7,11,27],uvicorn:27,v2:[27,28],v:[3,27],valu:[1,4,7,9,15,16,17,22,24,25],valueerror:17,van:[1,7,19,20,21,22,23],vanhulst:[7,19,20,21,22,23],vari:9,variabl:27,variat:[1,4,9,12],variou:28,vector:[1,4,12,21],verbos:[1,9],veri:[1,22],version:[0,3,4,9],via:[5,22,26,28],virtual:3,visit:8,vocab:1,vocabulari:[14,20],volum:[1,21,27],vri:[1,7,19,20,21,22,23],w2v:[1,4,9],w2v_1800_new:4,w2v_1800s_new:1,w2v_1810_new:4,w2v_1860s_new:1,w2v_:[1,9],w2v_ocr:[1,9],w2v_ocr_model:[1,9],w2v_ocr_pair:[1,4,12],w2v_ocr_path:[1,9],w2v_xxxxs_new:4,wa:[4,7,9,10,22],wai:[1,17],want:[1,3,4],warn:16,we:[0,1,3,4,5,15,17,19,22,25],web:4,webanno:16,weight:1,weight_decai:[1,10],well:[1,4,22],were:22,west:4,wget:3,what:22,whatev:22,when:[1,4,7,8,9,10,15,17,21,22],whenc:22,whenev:22,where:[1,4,9,12,13,15,17,19,22,24],wherea:[4,22],whereaft:22,wherebi:22,wherein:22,whereupon:22,wherev:22,whether:[1,4,7,8,9,10,17,22,23],which:[1,3,4,7,8,9,15,17,19,22,23,24,26,27],white:[15,16],whither:22,who:[4,22],whoever:22,whole:[1,22],whom:22,whose:[1,21,22,25],why:22,wiki2gaz:4,wiki2vec:4,wiki:[16,18],wiki_page_titl:18,wiki_titl:24,wikidata2wikipedia:18,wikidata:[1,2,5,7,8,9,12,16,17,18,24,25],wikidata_gazett:1,wikidata_id:[1,4],wikidata_norm:1,wikidata_to_ment:[9,12],wikidata_to_mentions_norm:1,wikidta_gazett:4,wikigaz_id:24,wikimapp:18,wikipedia2vec:25,wikipedia2wikidata:[16,18],wikipedia:[1,2,5,7,9,12,16,18,24],wikipedia_titl:16,wildcard:4,wilt:4,wilton:4,wiltshir:[1,4],wiltshrr:1,wise:1,wish:4,with_publ:[1,7],within:[9,22],without:[17,22],without_microtoponym:[1,7,8],withouttest:4,wk_cand:[8,25],wkdt_class:[1,8],wkdt_qid:[4,16],wkdtalt:[1,9],wmtop:[7,27,28],won:4,word1:4,word2:4,word2vec:[1,12],word:[1,2,5,10,12,15,16,17,21,22,23,25],work:[3,4,5,9],would:[1,18,19,22],wpubl:[7,27,28],wrap:[1,7],write:12,wv:[1,4],x:[1,28],xxxx:4,xz:3,yaml:[1,12],year:[4,7,16,17,19,20,21,22,23],yet:[15,17,22],yml:[26,27],york:[8,16,18],you:[0,1,3,4,8,22,27,28],your:[1,3,4,5,22,26,28],your_config_nam:27,your_host_url:27,yourself:22,yourselv:22,zenodo:4,zlib1g:3},titles:["Experiments and evaluation","The complete tour","Getting started","Installing T-Res","Resources and directory structure","T-Res: A Toponym Resolution Pipeline for Digitised Historical Newspapers","<code class=\"docutils literal notranslate\"><span class=\"pre\">geoparser</span></code> module","<code class=\"docutils literal notranslate\"><span class=\"pre\">geoparser.linking.Linker</span></code>","<code class=\"docutils literal notranslate\"><span class=\"pre\">geoparser.pipeline.Pipeline</span></code>","<code class=\"docutils literal notranslate\"><span class=\"pre\">geoparser.ranking.</span> <span class=\"pre\">Ranker</span></code>","<code class=\"docutils literal notranslate\"><span class=\"pre\">geoparser.recogniser.Recogniser</span></code>","Reference","<code class=\"docutils literal notranslate\"><span class=\"pre\">utils.deezy_processing</span></code> module","<code class=\"docutils literal notranslate\"><span class=\"pre\">utils.get_data</span></code> module","<code class=\"docutils literal notranslate\"><span class=\"pre\">utils</span></code> module","<code class=\"docutils literal notranslate\"><span class=\"pre\">utils.ner</span></code> module","<code class=\"docutils literal notranslate\"><span class=\"pre\">utils.preprocess_data</span></code> module","<code class=\"docutils literal notranslate\"><span class=\"pre\">utils.process_data</span></code> module","<code class=\"docutils literal notranslate\"><span class=\"pre\">utils.process_wikipedia</span></code> module","<code class=\"docutils literal notranslate\"><span class=\"pre\">utils.REL.entity_disambiguation</span></code> module","<code class=\"docutils literal notranslate\"><span class=\"pre\">utils.REL</span></code> module","<code class=\"docutils literal notranslate\"><span class=\"pre\">utils.REL.mulrel_ranker</span></code> module","<code class=\"docutils literal notranslate\"><span class=\"pre\">utils.REL.utils</span></code> module","<code class=\"docutils literal notranslate\"><span class=\"pre\">utils.REL.vocabulary</span></code> module","<code class=\"docutils literal notranslate\"><span class=\"pre\">utils.rel_e2e</span></code> module","<code class=\"docutils literal notranslate\"><span class=\"pre\">utils.rel_utils</span></code> module","Deploying the T-Res API","Deploying the T-Res API","Using the T-Res API"],titleterms:{"1":[0,1,4,27],"2":[0,1,4,27],"3":[0,1,27],"4":[0,1,27],A:5,The:1,an:1,api:[26,27,28],base:4,build:27,candid:1,commit:3,complet:1,compos:27,configur:27,contain:27,content:[2,5,6,11,14,20,26],csv:4,data:[0,4],dataset:[1,4],deezy_process:12,deezymatch:[1,4],deploi:[26,27],deploy:27,descript:1,digitis:5,directori:4,disambigu:[1,4],docker:27,embed:4,end:1,entiti:[1,4],entity2class:4,entity_disambigu:19,evalu:0,exist:1,experi:0,extern:0,from:1,gener:1,geopars:[6,7,8,9,10],get:2,get_data:13,given:1,histor:5,hoook:3,how:3,includ:1,indic:5,instal:3,instanti:1,json:4,levenshtein:1,link:7,linker:[1,7],load:1,mention:1,mentions_to_wikidata:4,mentions_to_wikidata_norm:4,model:1,modul:[6,12,13,14,15,16,17,18,19,20,21,22,23,24,25],mostpopular:1,mulrel_rank:21,multipl:27,ner:[1,15],newspap:5,noisi:4,obtain:0,option:1,output:1,pair:1,partialmatch:1,perfectmatch:1,pipelin:[1,5,8],poetri:3,pre:3,prepar:0,preprocess_data:16,process_data:17,process_wikipedia:18,project:3,pyenv:3,rank:9,ranker:[1,9],re:[3,5,26,27,28],recognis:[1,10],recognit:4,recommend:1,refer:11,rel:[19,20,21,22,23],rel_e2:24,rel_util:25,reldisamb:1,resolut:5,resourc:[0,1,4],retriev:1,run:0,scratch:1,set:4,start:2,step:1,string:1,structur:4,summari:4,system:3,t:[3,5,26,27,28],tabl:[2,5,6,11,14,20,26],toponym:[4,5],tour:1,train:[1,4],txt:4,updat:3,us:[1,3,28],util:[12,13,14,15,16,17,18,19,20,21,22,23,24,25],via:27,vocabulari:23,wikidata:4,wikidata_gazett:4,wikidata_to_mentions_norm:4,wikipedia:4,word2vec:4,word:4,your:27}})