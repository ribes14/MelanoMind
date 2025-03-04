import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.class_weight import compute_class_weight

from tensorflow import keras
from tensorflow.keras import layers, regularizers

# ---------------------------
# 1. Data Loading and Setup
# ---------------------------
train_df = pd.read_csv("data/new-train-metadata.csv", low_memory=False)
test_df = pd.read_csv("data/students-test-metadata.csv")

# Prepare target and align features
y = train_df["target"]
train_df = train_df.drop("target", axis=1)
test_columns = test_df.columns.tolist()
train_df = train_df[test_columns]

# Drop unnecessary columns
columns_to_drop = [
    "patient_id",
    "attribution",
    "anatom_site_general",
    "copyright_license",
]
train_df = train_df.drop(columns=columns_to_drop)
test_df = test_df.drop(columns=columns_to_drop)

# Store isic_id for submission and drop it from features
isic_test = test_df["isic_id"]
train_df = train_df.drop(["isic_id"], axis=1)
test_df = test_df.drop(["isic_id"], axis=1)

# Identify feature types
categorical_columns = train_df.select_dtypes(include=["object", "category"]).columns
numerical_columns = train_df.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()


# ---------------------------
# 2. Custom Age Transformer
# ---------------------------
class AgeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, division=90):
        self.division = division

    def fit(self, X, y=None):
        self.median_ = np.nanmedian(X.astype(float), axis=0)
        return self

    def transform(self, X):
        X = X.astype(float)
        X_imputed = np.where(np.isnan(X), self.median_, X)
        scaled_age = X_imputed / self.division
        age = X_imputed.flatten()
        menores_de_edad = ((age >= 0) & (age <= 17)).astype(float).reshape(-1, 1)
        jovenes = ((age >= 18) & (age <= 44)).astype(float).reshape(-1, 1)
        adulto_medio = ((age >= 45) & (age <= 59)).astype(float).reshape(-1, 1)
        adulto_mayor = ((age >= 60) & (age <= 74)).astype(float).reshape(-1, 1)
        anciano = ((age >= 75) & (age <= 90)).astype(float).reshape(-1, 1)
        return np.concatenate(
            [scaled_age, menores_de_edad, jovenes, adulto_medio, adulto_mayor, anciano],
            axis=1,
        )


# ---------------------------
# 3. Define Preprocessing Pipelines
# ---------------------------
# For numerical features, we now use RobustScaler.
num_pipeline = Pipeline(
    steps=[
        ("impute", IterativeImputer(initial_strategy="median")),
        ("scale", RobustScaler()),
    ]
)

age_pipeline = Pipeline(steps=[("age_transform", AgeTransformer(division=90))])

cat_pipeline = Pipeline(
    steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]
)

preprocesamiento = ColumnTransformer(
    transformers=[
        (
            "num",
            num_pipeline,
            [col for col in numerical_columns if col != "age_approx"],
        ),
        ("age", age_pipeline, ["age_approx"]),
        ("cat", cat_pipeline, categorical_columns),
    ]
)

# ---------------------------
# 4. Cross Validation Setup
# ---------------------------
# Split the data once for cross validation purposes.
X = train_df.copy()
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_val_auc = []

# Compute class weights once (using the full training target)
class_weights = compute_class_weight("balanced", classes=np.unique(y), y=y)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
print("Class weights:", class_weight_dict)


# ---------------------------
# 5. Define Model Function
# ---------------------------
def build_model(input_dim):
    model = keras.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(
                128, activation="relu", kernel_regularizer=regularizers.l2(0.001)
            ),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(
                64, activation="relu", kernel_regularizer=regularizers.l2(0.001)
            ),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="AUC")],
    )
    return model


# ---------------------------
# 6. Cross Validation Loop
# ---------------------------
fold = 1
for train_index, val_index in kf.split(X):
    print(f"\nStarting Fold {fold}")
    X_train_cv, X_val_cv = X.iloc[train_index], X.iloc[val_index]
    y_train_cv, y_val_cv = y.iloc[train_index], y.iloc[val_index]

    # Fit preprocesamiento on the training split and transform both splits
    preprocesamiento.fit(X_train_cv)
    X_train_cv_transformed = preprocesamiento.transform(X_train_cv)
    X_val_cv_transformed = preprocesamiento.transform(X_val_cv)

    input_dim = X_train_cv_transformed.shape[1]
    model = build_model(input_dim)

    # Define callbacks with a slightly lower patience for quicker convergence checking
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_AUC", patience=5, mode="max", restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_AUC", factor=0.5, patience=3, mode="max"
        ),
    ]

    history = model.fit(
        X_train_cv_transformed,
        y_train_cv.to_numpy(),
        validation_data=(X_val_cv_transformed, y_val_cv.to_numpy()),
        epochs=50,
        batch_size=2048,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=0,
    )

    best_val_auc = max(history.history["val_AUC"])
    cv_val_auc.append(best_val_auc)
    print(f"Fold {fold} best val AUC: {best_val_auc:.4f}")
    fold += 1

print("\nAverage CV AUC:", np.mean(cv_val_auc))

# ---------------------------
# 7. Final Training on Full Data
# ---------------------------
# Retrain the preprocessor on the full training data.
preprocesamiento.fit(X)
X_full_transformed = preprocesamiento.transform(X)
X_test_transformed = preprocesamiento.transform(test_df)

# Retrieve feature names (for reference if needed)
num_feature_names = [col for col in numerical_columns if col != "age_approx"]
age_feature_names = [
    "age_approx_scaled",
    "menores_de_edad",
    "jovenes",
    "adulto_medio",
    "adulto_mayor",
    "anciano",
]
cat_feature_names = (
    preprocesamiento.named_transformers_["cat"]
    .named_steps["encode"]
    .get_feature_names_out(categorical_columns)
)
feature_names = list(num_feature_names) + age_feature_names + list(cat_feature_names)

X_full_df = pd.DataFrame(X_full_transformed, columns=feature_names)
X_test_df = pd.DataFrame(X_test_transformed, columns=feature_names)

# Determine best epochs from CV (here we could average best epochs from folds, but for simplicity we use early stopping during training)
input_dim = X_full_df.shape[1]
final_model = build_model(input_dim)

# Use callbacks again to prevent overfitting
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_AUC", patience=10, mode="max", restore_best_weights=True
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_AUC", factor=0.5, patience=5, mode="max"
    ),
]

# Split a small validation set for final training monitoring
X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(
    X_full_df, y, test_size=0.1, random_state=42, stratify=y
)

history_final = final_model.fit(
    X_train_final,
    y_train_final.to_numpy(),
    validation_data=(X_val_final, y_val_final.to_numpy()),
    epochs=100,
    batch_size=2048,
    class_weight=class_weight_dict,
    callbacks=callbacks,
    verbose=1,
)

# ---------------------------
# 8. Evaluate and Predict
# ---------------------------
# Plot training history for final model
plt.plot(history_final.history["AUC"], label="Training AUC")
plt.plot(history_final.history["val_AUC"], label="Validation AUC")
plt.xlabel("Epoch")
plt.ylabel("AUC")
plt.legend()
plt.show()

# Predict on test set
y_pred_proba = final_model.predict(X_test_df)

# Create submission file
results = pd.DataFrame({"isic_id": isic_test, "target": y_pred_proba.flatten()})
results.to_csv("predicciones.csv", index=False)
print("Predictions exported successfully.")
print(results.head())
