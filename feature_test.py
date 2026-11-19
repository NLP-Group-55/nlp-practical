# we will be using textstat as a way to otain some descriptive features of text;
# we encourage you to look for things that are meaningful to your own task/dataset
# this is just to demo the kind of thing you can do
import textstat

import pandas as pd

table = []
systems = dict()
items = dict()

# each model uses a certain tokenizer, and we don't want to accidentally use the wrong one
name_to_tokenizer = {
    'embed-and-pool': tokenizer_for_bow,
    'lstm': tokenizer_for_rnn
}

for model_family, model_data in my_classifiers.items(): # for all our trained classifiers
    for run, (seed, run_data) in enumerate(zip(my_seeds, model_data)):
        # we get the devtest set that we kep aside for each run
        # both texts and labels
        run_texts = [text for text, label in my_test_sets[run]['devtest']]
        run_labels = np.array([label for text, label in my_test_sets[run]['devtest']], dtype=int)

        for fold in range(num_folds):
            # load our model
            trained_nn = part2.experiment.load(f"results/{model_family}/seed-{seed}/fold-{fold}/model", trained_emb_model)
            # test it on the devtest
            logprobs = part2.experiment.predict_logprobs(
                trained_nn,
                name_to_tokenizer[model_family],
                run_texts,
                64,
                'cpu'
            )
            preds = np.argmax(logprobs, -1)

            # then featurize the texts and compute the correctness flags
            for i in range(len(preds)):

                # find an int id for each system
                sid = systems.get(model_family, None)
                if sid is None:
                    sid = len(systems)
                    systems[model_family] = sid
                # find an int id for each item
                item = my_test_sets[run]['devtest_ids'][i]
                item_id = items.get(item, None)
                if item_id is None:
                    item_id = len(items)
                    items[item] = item_id

                # text, pred, and label
                x = run_texts[i]
                y_ = preds[i]
                y = run_labels[i]

                # put it all in a table, along with some features
                table.append([
                    model_family,
                    item,
                    sid,
                    item_id,
                    run,
                    fold,
                    y,
                    y_,
                    int(y == y_), # correctness flag
                    # some features
                    textstat.lexicon_count(x),
                    textstat.flesch_reading_ease(x),
                    textstat.reading_time(x),
                    textstat.flesch_kincaid_grade(x)
                ])

df = pd.DataFrame(table, columns=['system', 'item', 'system_id', 'item_id', 'run', 'fold',  'label', 'pred', 'correct', 'lexicon_count', 'flesch_reading_ease', 'reading_time', 'flesch_kincaid_grade'])
df
import numpy as np

def vocabulary_coverage(text, tokenizer, embedding_model):
    tokens = tokenizer.tokens(text)
    if not tokens:
        return 0.0
    return sum(token in embedding_model for token in tokens) / len(tokens)


# Match each devtest item to its text.
text_by_run_item = {}
for run_idx, test_set in enumerate(my_test_sets):
    texts = [text for text, label in test_set["devtest"]]
    item_ids = test_set["devtest_ids"]

    for item_id, text in zip(item_ids, texts):
        text_by_run_item[(run_idx, item_id)] = text


# Add vocabulary coverage to the existing dataframe.
df["vocab_coverage"] = [
    vocabulary_coverage(
        text_by_run_item[(int(run), item)],
        name_to_tokenizer[system],
        trained_emb_model,
    )
    for system, item, run in zip(df["system"], df["item"], df["run"])
]


# Build the BDA feature matrix.
prediction_features = part2.analysis.one_hot(
    df["pred"].to_numpy(),
    len(agnews_outcome_space),
)

continuous_cols = [
    "lexicon_count",
    "flesch_reading_ease",
    "reading_time",
    "flesch_kincaid_grade",
    "vocab_coverage",
]

X = df[continuous_cols].to_numpy(dtype=float)
feature_means = X.mean(axis=0)
feature_stds = X.std(axis=0)
feature_stds[feature_stds == 0] = 1.0
X_standardized = (X - feature_means) / feature_stds

features = np.hstack([prediction_features, X_standardized])

mcmc = part2.analysis.fit_bda_model(
    num_systems=len(systems),
    sys_id=df["system_id"].to_numpy(),
    features=features,
    correctness=df["correct"].to_numpy(),
    num_chains=2,
)

mcmc.print_summary()