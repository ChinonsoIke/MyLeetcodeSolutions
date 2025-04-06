using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace ConsoleApp1
{
    internal class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine(PartitionLabels("ababcbacadefegdehijhklij"));
        }

        public static IList<int> PartitionLabels(string s)
        {
            var sb = new StringBuilder();
            var prevSb = "";
            var list = new List<int>();

            while (s.Length != 0)
            {
                if (sb.Length == 0)
                {
                    int last = s.LastIndexOf(s[0]);
                    sb.Append(s.Substring(0, s.LastIndexOf(s[0])+1));
                    s = s.Substring(last+1);
                }
                else
                {
                    if (sb.ToString().Contains(s[0]))
                    {
                        sb.Append(s[0]);
                        s = s.Substring(1);                    }
                    else
                    {
                        if (prevSb.Contains(s[0]))
                        {
                            list[list.Count - 1]++;
                        }
                        list.Add(sb.Length);
                        prevSb = sb.ToString();
                        sb.Clear();
                    }
                }
            }

            return list;
        }

        public static string isBalanced(string s)
        {
            //if open bracket,add to stack
            //if not, check if top of stack matches
            //if not return false, else pop

            var stack = new Stack<char>();
            foreach (char c in s)
            {
                if (c == '(' || c == '[' || c == '{') stack.Push(c);
                else
                {
                    if (c == ')')
                    {
                        if (stack.Peek() == '(') stack.Pop();
                        else return "NO";
                    }
                    if (c == '}')
                    {
                        if (stack.Peek() == '{') stack.Pop();
                        else return "NO";
                    }
                    if (c == ']')
                    {
                        if (stack.Peek() == '[') stack.Pop();
                        else return "NO";
                    }
                }
            }

            return "YES";
        }

        public static int twoStacks(int maxSum, List<int> a, List<int> b)
        {
            int count = 0, i = 0, j = 0, sum = 0;

            while (i < a.Count && j < b.Count)
            {
                if (a[i] <= b[j])
                {
                    sum += a[i];
                    count++;
                    i++;
                }
                else
                {
                    sum += b[j];
                    count++;
                    j++;
                }

                if (sum > maxSum)
                {
                    count--;
                    break;
                }
            }

            while (i < a.Count && sum < maxSum)
            {
                sum += a[i];
                count++;
                i++;

                if (sum > maxSum)
                {
                    count--;
                    break;
                }
            }

            while (j < b.Count && sum < maxSum)
            {
                sum += b[j];
                count++;
                j++;

                if (sum > maxSum)
                {
                    count--;
                    break;
                }
            }

            return count;
        }

        public static int[] DailyTemperatures(int[] temperatures)
        {
            int n = temperatures.Length, maxIndex = n - 1;
            int[] days = new int[n];

            var stack = new Stack<int>();
            stack.Push(0);

            for (int i = 1; i<n; i++)
            {
                while (temperatures[i] > stack.Peek())
                {
                    int x = stack.Peek();
                    days[x] = i - stack.Peek();
                    stack.Pop();
                }
                stack.Push(i);
            }

            return days;
        }

        public int[] DailyTemperatures2(int[] temperatures)
        {
            int n = temperatures.Length, maxIndex = n - 1;
            int[] days = new int[n];

            for (int i = n - 2; i >= 0; i--)
            {
                if (temperatures[i] >= temperatures[maxIndex])
                {
                    maxIndex = i;
                    days[i] = 0;
                }
                else
                {
                    for (int j = i + 1; j <= maxIndex; j++)
                    {
                        if (temperatures[j] > temperatures[i])
                        {
                            days[i] = j - i;
                            break;
                        }
                    }
                }
            }

            return days;
        }

        public static int MaxArea(int[] height)
        {
            int maxHeight = height[0], maxArea = 0;
            for (int i = 1; i < height.Length; i++)
            {
                int min = Math.Min(height[i], maxHeight);
                maxArea = Math.Max(maxArea, min * min);
                maxHeight = Math.Max(maxHeight, height[i]);
            }

            return maxArea;
        }

        public static void Merge(int[] a, int[] b)
        {
            // a and b are sorted in ascending order
            // b.length = m, a.length = m+n
            // a contains n elements and m elements filled with zeroes
            // eg a = { 1, 4, 6, 7,0,0,0 } b = { 2,3,8 }
            for (int i = a.Length-b.Length, j=0; i < a.Length; i++,j++)
            {
                int k = i;
                while (b[j] < a[k-1] && k > 0)
                {
                    a[k] = a[k-1];
                    k--;
                }
                a[k] = b[j];
            }
        }

        public static int MaxRepeating(string sequence, string word)
        {
            int i = sequence.IndexOf(word);
            if (i < 0) return 0;
            int h1 = 0, h2 = 0;
            //sequence = sequence.Substring(i+word.Length);

            while (true)
            {
                h2++;
                if (i + word.Length*2 <= sequence.Length && sequence.Substring(i+word.Length, word.Length) == word)
                {
                    sequence = sequence.Substring(word.Length);
                    continue;
                }
                else
                {
                    sequence = sequence.Substring(i+word.Length/2);
                    if (h2 > h1) h1 = h2;
                    h2 = 0;
                    i = sequence.IndexOf(word);
                    if (i < 0) return h1;
                }
            }
        }

        public static bool IsSubsequence(string s, string t)
        {
            if (String.IsNullOrEmpty(s)) return true;
            if (String.IsNullOrEmpty(t)) return false;

            int[] indexes = new int[s.Length];
            indexes[0] = t.IndexOf(s[0]);
            if (indexes[0] < 0) return false;
            for (int i = 1; i < indexes.Length; i++)
            {
                //t = indexes[i - 1] == t.Substring(1) : t.Substring(indexes[i - 1]);
                indexes[i] = t.IndexOf(s[i]);
                if (indexes[i] < 0) return false;
            }
            return true;
        }

        public static void SearchDAC(int[][] M, int start, int end, int key)
        {
            if(start > end)
            {
                Console.WriteLine("Not Found");
                return;
            }
            int mid = (start + end) / 2;
            if(key >= M[mid][0] && key <= M[mid][M[mid].Length - 1])
            {
                int bin = BinarySearch(M[mid], 0, M[mid].Length - 1, key);
                if (bin > -1)
                {
                    Console.WriteLine($"{mid},{bin}");
                    return;
                }
            }
            else if(key < M[mid][0])
                SearchDAC(M, start, mid-1, key);
            else SearchDAC(M, mid+1, end, key);            
        }

        public static int BinarySearch(int[] arr, int start, int end, int key)
        {
            if(start > end) return -1;
            int mid = (start + end) / 2;
            if(key == arr[mid]) return mid;
            if(key < arr[mid]) return BinarySearch(arr, start, mid-1, key);
            else return BinarySearch(arr, mid+1, end, key);
        }

        public static List<int> minPartition(int N)
        {
            //Your code here
            List<int> list = new List<int> { 2000, 500, 200, 100, 50, 20, 10, 5, 2, 1 };
            List<int> list2 = new List<int>();

            for (int i = 0; i < list.Count; i++)
            {
                if (N >= list[i])
                {
                    int num = N / list[i];
                    N = N % list[i];
                    for (int j = 1; j <= num; j++) list2.Add(list[i]);
                }
                if (N == 0) break;
            }

            return list2;
        }

        public int LongestPalindrome(string s)
        {
            char[] chars = s.ToCharArray();
            var dict = new Dictionary<char,int>();
            bool oddFound = false;
            int total = 0;

            for (int i = 0; i < chars.Length; i++)
            {
                if(dict.ContainsKey(chars[i])) dict[chars[i]]++;
                else dict[chars[i]] = 1;
            }

            foreach (int i in dict.Values)
            {
                if (i % 2 == 0) total += i;
                else
                {
                    if (!oddFound)
                    {
                        total += i;
                        oddFound = true;
                    }
                    else total += i - i%2;
                }
            }

            return total;
        }

        public int splitNum(int num)
        {
            char[] arr = num.ToString().ToCharArray();
            Array.Sort(arr);
            StringBuilder sb1 = new StringBuilder(), sb2 = new StringBuilder();

            for (int i = 0; i < arr.Length; i+=2)
            {
                sb1.Append(arr[i]);
                if (i < arr.Length - 1) sb2.Append(arr[i + 1]);
            }

            return int.Parse(sb1.ToString()) + int.Parse(sb2.ToString());
        }

        public static int[] AnswerQueries(int[] nums, int[] queries)
        {
            Array.Sort(nums);
            int[] answers = new int[queries.Length];

            for (int i = 0; i < queries.Length; i++)
            {
                int count = 0, total = 0;
                for (int j = 0; j < nums.Length; j++)
                {
                    if (total + nums[j] <= queries[i])
                    {
                        total += nums[j];
                        count++;
                    }else break;
                }
                answers[i] = count;
            }

            return answers;
        }

        static int countWays(int n)
        {

            // base case
            if (n < 0)
                return 0;
            if (n == 0)
                return 1;

            return countWays(n - 1) + countWays(n - 3) + countWays(n - 5);
        }

        static int cutRod(int[] price)
        {
            int n = price.Length;
            int[] dp = new int[n + 1];

            // Find maximum value for all 
            // rod of length i.
            for (int i = 1; i <= n; i++)
            {
                for (int j = 1; j <= i; j++)
                {
                    dp[i] = Math.Max(dp[i], price[j - 1] + dp[i - j]);
                }
            }

            return dp[n];
        }

        //public static int FibonacciDP(int n)
        //{
        //    if (!map.ContainsKey(n))
        //    {
        //        map[n] = FibonacciDP(n-2) + FibonacciDP(n-1);
        //    }
        //    return map[n];
        //}

        public static bool CanFinish(int numCourses, int[][] prerequisites)
        {
            // build adjacency matrix
            // do bsf

            // if(prerequisites.Length == 1 && prerequisites[0][0] != prerequisites[0][1]) return true;
            int[][] adjMatrix = new int[numCourses][];
            for (int i = 0; i < numCourses; i++)
            {
                adjMatrix[i] = new int[numCourses];
            }

            for (int i = 0; i < prerequisites.Length; i++)
            {
                int u = prerequisites[i][0], v = prerequisites[i][1];
                if (u == v) return false;
                adjMatrix[u][v] = 1;
                if (adjMatrix[u][v] == 1 && adjMatrix[v][u] == 1) return false;
            }

            bool[] visited = new bool[numCourses];
            bool[] isInCallStack = new bool[numCourses];

            for (int i = 0; i < numCourses; i++)
            {
                if (!visited[i] && IsCyclic(adjMatrix, i, visited, isInCallStack)) return false;
            }
            // Queue<int> q = new Queue<int>();
            // q.Enqueue(0);
            // visited[0] = true;

            // while(q.Count > 0){
            //     var cur = q.Dequeue();
            //     if(adjMatrix[cur][cur] == 1) return false;

            //     for(int i=0; i<numCourses; i++){
            //         if(adjMatrix[cur][i] == 1 && !visited[i]){
            //             if(adjMatrix[i][cur] == 1) return false;
            //             q.Enqueue(i);
            //             visited[i] = true;
            //         }
            //     }
            // }

            return true;
        }

        public static bool IsCyclic(int[][] adjMatrix, int startIndex, bool[] visited, bool[] isInCallStack)
        {
            if (isInCallStack[startIndex]) return true;

            isInCallStack[startIndex] = true;
            visited[startIndex] = true;

            for (int i = 0; i < adjMatrix.Length; i++)
            {
                if (adjMatrix[startIndex][i] == 1 && IsCyclic(adjMatrix, i, visited, isInCallStack)) return true;
                // if(isInCallStack[i]) return true;
            }

            isInCallStack[startIndex] = false;
            return false;
        }

        public int FindJudge(int n, int[][] trust)
        {
            var list = new List<int>();
            for (int i = 1; i < trust.Length; i++)
            {
                var intersect = trust[i].Intersect(trust[i - 1]);
                if (intersect == null) return -1;
                list = intersect.ToList();
            }

            for (int i = 0; i < trust.Length; i++)
            {
                if (trust[i][0] == list[0]) return -1;
            }

            return list[0];
        }

        public static string balancedSums(List<int> arr)
        {
            if (arr.Count == 1) return "YES";
            if (arr.Count == 2) return "NO";

            int right = 0;
            for (int i = 1; i < arr.Count - 1; i++)
            { //2 0 0 0
                right += arr[i - 1];

                int left = 0;
                for (int k = i + 1; k < arr.Count; k++)
                {
                    left += arr[k];
                }

                if (right == left) return "YES";
            }

            return "NO";
        }

        public static string Reverse (string s)
        {
            if(s.Length == 1) return s;

            return s[s.Length - 1] + Reverse(s.Substring(0, s.Length-1)); //ever
            // r + reverse(eve) = reve
            // e + reverse(ev) = eve
            // v + reverse(e) = ve
            // e
        }

        public static bool ValidPath(int n, int[][] edges, int source, int destination)
        {
            // build adjacency list
            // do bfs

            // if(edges.Length < 1) return true;
            Queue<int> queue = new Queue<int>();
            bool[] visited = new bool[n];
            List<List<int>> adjMatrix = new List<List<int>>();
            //Array.Fill(adjMatrix, new int[n]);

            for (int i = 0; i < n; i++)
            {
                adjMatrix.Add(new List<int>());
            }

            for (int i = 0; i < edges.Length; i++)
            {
                int u = edges[i][0], v = edges[i][1];
                adjMatrix[u].Add(v);
                // adjMatrix[v][u] = 1;
            }
            visited[source] = true;
            queue.Enqueue(source);

            while (queue.Count > 0)
            {
                int cur = queue.Dequeue();
                if (cur == destination || cur == destination) return true;

                foreach (int i in adjMatrix[cur])
                {
                    if (!visited[i])
                    {
                        queue.Enqueue(i);
                        visited[i] = true;
                    }
                }
            }

            return false;
        }
    }
}
