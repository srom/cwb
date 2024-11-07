#!/bin/bash

calculate_elapsed_time() {
    local start=$1
    local end=$2
    local elapsed_ns=$((end - start))  # elapsed time in nanoseconds
    local elapsed_ms=$((elapsed_ns / 1000000))  # convert to milliseconds
    local ms=$((elapsed_ms % 1000))  # milliseconds
    local total_seconds=$((elapsed_ms / 1000))  # total seconds
    local hours=$((total_seconds / 3600))
    local minutes=$(( (total_seconds % 3600) / 60 ))
    local seconds=$((total_seconds % 60))

    local output=""

    if [ $hours -gt 0 ]; then
        output="${hours}h "
    fi

    if [ $minutes -gt 0 ] || [ $hours -gt 0 ]; then
        output="${output}${minutes}m "
    fi

    if [ $seconds -gt 0 ] || [ $minutes -gt 0 ] || [ $hours -gt 0 ]; then
        output="${output}${seconds}s "
    fi

    if [ $total_seconds -lt 1 ]; then
        output="${ms}ms"
    elif [ $ms -gt 0 ]; then
        output="${output}${ms}ms"
    fi

    echo $output
}

##
# Remove sequences with more too many gaps.
##
remove_sequences_with_too_many_gaps() {
    local in=$1
    local out=$2
    local threshold=$3

    awk -v threshold="${threshold}" '
    /^>/ { 
        # For header lines (starting with ">"):
        # Check if the current sequence (if any) has X% or fewer gaps
        if (seqlen > 0 && gaps / seqlen <= threshold) {
            print header   # Print the header
            print seq      # Print the sequence
        }
        # Reset variables for the next sequence
        header=$0       # Store the current header
        seq=""          # Reset sequence
        seqlen=0        # Reset sequence length
        gaps=0          # Reset gap count
        next            # Skip to the next line
    }

    {
        # For sequence lines:
        seq=seq $0           # Append the line to the sequence
        seqlen+=length($0)   # Increment total sequence length
        gaps+=gsub(/-/, "-", $0) # Count gaps in this line and add to total
    }

    END {
        # After processing all lines:
        # Check and print the last sequence (if it passes the gap check)
        if (seqlen > 0 && gaps / seqlen <= threshold) {
            print header   # Print the header
            print seq      # Print the sequence
        }
    }
    ' "${in}" > "${out}"
}

split_fasta() {
    # Check for correct number of arguments
    if [ "$#" -ne 3 ]; then
        echo "Usage: split_fasta <input_fasta> <output_first_half> <output_second_half>"
        return 1
    fi

    input_file=$1
    first_half_output=$2
    second_half_output=$3

    # AWK script embedded within the function
    awk '
    /^>/ {
        if (NR > 1) {
            print_half_sequences()
        }
        header = $0
        seq = ""
        next
    }

    {
        seq = seq $0
    }

    function print_half_sequences() {
        mid = int(length(seq) / 2)
        first_half = substr(seq, 1, mid)
        second_half = substr(seq, mid + 1)
        print header > "'"$first_half_output"'"
        print first_half > "'"$first_half_output"'"
        print header > "'"$second_half_output"'"
        print second_half > "'"$second_half_output"'"
    }

    END {
        print_half_sequences()
    }
    ' "$input_file"
}
