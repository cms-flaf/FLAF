#!/bin/bash

if [[ $1 == "--dry-run" ]]; then
    DRY_RUN=true
    DRY_RUN_PREFIX="(dry run) "
else
    DRY_RUN=false
    DRY_RUN_PREFIX=""
fi

IFS_PREV=$IFS
IFS=$'\n'
declare -a PYTHON_FILES=()
declare -a CPP_FILES=()
declare -a YAML_FILES=()
for file in $(git log --name-only --pretty="" origin/main..HEAD | sort | uniq); do
    if [ ! -f "$file" ]; then
        continue
    fi
    if [[ $file == *.py ]]; then
        PYTHON_FILES+=("$file")
    elif [[ $file == *.cpp || $file == *.h || $file == *.hpp || $file == *.cc ]]; then
        CPP_FILES+=("$file")
    elif [[ $file == *.yaml || $file == *.yml || $file == .yamllint ]]; then
        YAML_FILES+=("$file")
    fi
done

if [ ${#PYTHON_FILES[@]} -gt 0 ]; then
    echo "${DRY_RUN_PREFIX}Applying Python formatting to: ${PYTHON_FILES[@]}"
    if [ "$DRY_RUN" = true ]; then
        black --check --diff "${PYTHON_FILES[@]}"
    else
        black "${PYTHON_FILES[@]}"
    fi
fi

if [ ${#CPP_FILES[@]} -gt 0 ]; then
    if [ -f "$ANALYSIS_PATH/.clang-format" ]; then
        clang_format_style="$ANALYSIS_PATH/.clang-format"
    elif [ -f "$FLAF_PATH/.clang-format" ]; then
        clang_format_style="$FLAF_PATH/.clang-format"
    else
        echo "No clang-format configuration file found."
        exit 1
    fi

    echo "${DRY_RUN_PREFIX}Applying C++ formatting to: ${CPP_FILES[@]}"
    if [ "$DRY_RUN" = true ]; then
        clang-format --dry-run --Werror --style "file:${clang_format_style}" "${CPP_FILES[@]}"
    else
        clang-format -i --Werror --style "file:${clang_format_style}" "${CPP_FILES[@]}"
    fi
fi

if [ ${#YAML_FILES[@]} -gt 0 ]; then
    if [ -f "$ANALYSIS_PATH/.yamllint" ]; then
        yamllint_config="$ANALYSIS_PATH/.yamllint"
    elif [ -f "$FLAF_PATH/.yamllint" ]; then
        yamllint_config="$FLAF_PATH/.yamllint"
    else
        echo "No yamllint configuration file found."
        exit 1
    fi
    echo "Checking YAML formatting for: ${YAML_FILES[@]}"
    yamllint -s -c ${yamllint_config} "${YAML_FILES[@]}"
fi

IFS=$IFS_PREV
