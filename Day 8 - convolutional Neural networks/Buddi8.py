# Import the necessary modules or libraries
import matplotlib.pyplot as plt

# Function to perform convolution on two strings
def convolution(s1, s2):
    # Split the strings into lists of words
    s1 = list(s1.split())
    s2 = list(s2.split())
    
    # Initialize lists to store exact and partial matches
    exact = []
    partial = []

    # Loop over each element in the first string's list
    for i in range(len(s1)):
        e = 0  # Counter for exact matches
        p = 0  # Counter for partial matches
        j = i + 1
        while i != -1:
            # Create a reversed sublist of s2 up to the current position
            s3 = s2[0:j][::-1]
            # Check if the current word in s1 matches the corresponding word in the reversed s3
            if s1[len(s1)-1-i] == s3[i]:
                e += 1
            # Check if the current word in s2 is in s1
            if s2[i] in s1:
                p += 1
            i -= 1
        exact.append(e)
        partial.append(p)

    # Loop over the remaining elements in the first string's list
    for i in range(len(s1) - 1):
        e = 0  # Counter for exact matches
        p = 0  # Counter for partial matches
        while i < len(s1) - 1:
            # Check if the current word in s1 matches the next word in s2
            if s1[i] == s2[i + 1]:
                e += 1
            # Check if the next word in s2 is in s1
            if s2[i + 1] in s1:
                p += 1
            i += 1
        exact.append(e)
        partial.append(p)

    # Print the exact and partial match lists
    print(exact, partial)
    
    # Create a list of positions for plotting
    positions = list(range(-len(s1) + 1, len(s2)))
    print(positions)
    
    return positions, exact, partial

# Define two example strings
s1 = "he is a bad boy"
s2 = "I watch bad boy 2"

# Perform convolution on the two strings
positions, exact, partial = convolution(s1, s2)

# Plot the results
plt.plot(positions, exact, label='Exact Matches', marker='o')
plt.plot(positions, partial, label='Partial Matches', marker='x')
plt.xlabel('Position')
plt.ylabel('Matches')
plt.title('Convolution of Two Strings')
plt.figtext(0.5, 0.01, "This graph represents the Convolution of two strings. Here the peak is maximum where the strings have more exact matches", ha="center", fontsize=10, bbox={"facecolor":"lightgrey", "alpha":0.5, "pad":5})
plt.legend()
plt.show()