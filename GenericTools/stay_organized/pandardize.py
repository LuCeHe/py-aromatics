

def simplify_col_names(df):
    df = df.rename(
        columns={
            # 'val_sparse_categorical_accuracy': 'val_zacc',
            'val_sparse_mode_accuracy': 'val_macc',
            'sparse_categorical_accuracy_test': 'test_macc', 'val_perplexity': 'val_ppl',
            'perplexity_test': 'test_ppl',
            't_sparse_mode_accuracy': 't_macc', 't_perplexity': 't_ppl',
            'v_sparse_mode_accuracy': 'v_macc', 'v_perplexity': 'v_ppl',
            'val_sparse_categorical_crossentropy': 'val_xnt',
            'val_sparse_categorical_accuracy': 'val_acc',
        },
        inplace=True
    )

