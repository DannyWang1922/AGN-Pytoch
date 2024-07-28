python run_clf.py --config data/sst2/sst2.json
python run_clf.py --config data/sst2/sst2.bert.json

python run_ner.py --config data/ner/conll2003_AGN.json
python run_ner.py --config data/ner/conll2003_bert.json

python run_ner.py --config data/ner/conll2003_AGN_ae_sigmoid.json
python run_ner.py --config data/ner/conll2003_AGN_vae_sigmoid.json
python run_ner.py --config data/ner/conll2003_AGN_none_sigmoid.json

python run_ner.py --config data/ner/conll2003_AGN_ae_softmax.json
python run_ner.py --config data/ner/conll2003_AGN_vae_softmax.json
python run_ner.py --config data/ner/conll2003_AGN_none_softmax.json