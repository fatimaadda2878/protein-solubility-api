# Mettre à jour un dépôt GitHub existant avec Git Bash (Windows)

> Ce paquet contient le **code V7 et sa documentation GitHub**. Il ne contient **aucun checkpoint, dataset, token ou rapport métier définitif**.

## 1. Préparer les dossiers

1. Télécharger `protein_solubility_GitHub_V7_FINAL.zip` puis **Extraire tout** sous Windows (dans Téléchargements par exemple).
2. Ouvrir **Git Bash**.
3. Se rendre dans le **dépôt Git local existant**, celui qui possède déjà un dossier `.git`.

```bash
cd "/c/Users/VOTRE_SESSION/Documents/VOTRE_DEPOT_GITHUB"
git status
git remote -v
git branch --show-current
```

Si le projet n'est **pas encore cloné** sur cet ordinateur :

```bash
cd "/c/Users/VOTRE_SESSION/Documents"
git clone https://github.com/UTILISATEUR/NOM_DU_DEPOT.git
cd NOM_DU_DEPOT
```

Remplacer ces chemins fictifs par les vrais. Ne pas exécuter `git init` dans un dépôt déjà cloné.

## 2. Récupérer les dernières modifications

```bash
git pull --ff-only
```

Si Git indique des modifications locales non commitées, **ne pas les écraser** : vérifier `git status` et sauvegarder/committer d'abord.

## 3. Copier les nouveaux fichiers dans le dépôt

Exemple si le ZIP extrait donne `C:\Users\VOTRE_SESSION\Downloads\protein_solubility_GitHub_V7_FINAL` :

```bash
cp -a "/c/Users/VOTRE_SESSION/Downloads/protein_solubility_GitHub_V7_FINAL/." ./
```

Cela met à jour les fichiers du même nom, en conservant `.git`. Les documents d'anciennes versions éventuellement présents dans ton dépôt ne sont pas supprimés automatiquement ; examine-les pour éviter toute confusion.

## 4. Vérifier avant publication

```bash
git status --short
git diff --stat
git ls-files | grep -Ei '(\.pt$|\.joblib$|\.ckpt$|\.env$|token|secret|\.fasta$|\.zip$)' || true
```

La dernière commande permet de **repérer** d'éventuels fichiers privés **déjà suivis**, auxquels `.gitignore` ne s'applique pas rétroactivement.

Si un artefact privé est déjà suivi :

```bash
git rm --cached "CHEMIN_DU_FICHIER_PRIVE"
```

Ne pas utiliser cette commande sans remplacer le chemin fictif. Si un token a déjà été publié, le **révoquer/renouveler** même après suppression du fichier.

## 5. Commit et push

```bash
git add -A
git diff --cached --stat
git diff --cached --name-only
git status
```

Avant le commit, **vérifie visuellement les fichiers préparés** : pas de poids, données ni secrets. Pour retirer un fichier de la sélection : `git restore --staged "CHEMIN"`.

```bash
git commit -m "docs: finaliser le projet solubilite proteique V7"
git push origin "$(git branch --show-current)"
```

Si le dépôt est sur la branche `main`, le push ciblera `main` automatiquement. Si la branche est différente, le push cible la branche courante.

## 6. Contrôle final

- Ouvrir GitHub dans le navigateur et vérifier le README V7, les documents `docs/`, les scripts et les tests.
- Ouvrir le Space Hugging Face ; **il reste indépendant du push GitHub** tant qu'aucune synchronisation GitHub → Space n'est configurée.
- Ne pas relancer l'entraînement, ni remplacer les fichiers du dépôt Model privé.

## En cas d'erreur Git

- `fatal: not a git repository` : mauvais dossier ; se positionner dans le dépôt cloné.
- `Author identity unknown` : configurer `git config --global user.name "..."` et `git config --global user.email "..."` avec l'identité voulue.
- `rejected (non-fast-forward)` : faire `git pull --rebase` **après sauvegarde des changements locaux**, puis retenter le push.
- `nothing to commit` : rien n'a changé ou tu as copié dans le mauvais dossier.
- Échec d'authentification : utiliser Git Credential Manager ou la connexion GitHub HTTPS appropriée ; ne jamais mettre un token en clair dans la commande.
