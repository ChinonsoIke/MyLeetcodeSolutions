using Grpc.Net.Client;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml.Linq;

namespace ConsoleApp1
{
    internal class Program
    {
        public record struct rNode(int i, int j);
        static async Task Main(string[] args)
        {
            var app = new Program();
            //TreeNode tree1 = app.BuildTree([1, 2, 2, 3, 3, null, null, 4, 4]);
            //int a = 1011_0111, b = ~a;
            Console.WriteLine(app.FindOrder(4, [[1, 0], [2, 0], [3, 1], [3, 2]]));
        }

        public int[] FindOrder(int numCourses, int[][] prerequisites)
        {
            //build adjacency list
            //do dfs
            //if a cycle is found return false

            var adjList = new Dictionary<int, List<int>>();
            int[] visited = new int[numCourses];
            var list = new List<int>();

            for (int i = 0; i < numCourses; i++)
            {
                adjList.Add(i, new List<int>());
            }
            foreach (int[] edge in prerequisites)
            {
                if (edge[0] == edge[1]) return [];
                adjList[edge[1]].Add(edge[0]);
            }

            var stack = new Stack<int>();

            // graph could be disconnected
            while (Array.IndexOf(visited, 0) > -1)
            {
                stack.Push(Array.IndexOf(visited, 0));
                visited[Array.IndexOf(visited, 0)] = 1;

                while (stack.Count > 0)
                {
                    int course = stack.Peek();

                    bool found = false;
                    foreach (int c in adjList[course])
                    {
                        if (stack.Contains(c)) return [];

                        if (visited[c] == 0)
                        {
                            found = true; // we've found a neighbor to add to stack
                            visited[c] = 1;
                            stack.Push(c);
                            break;
                        }
                    }

                    if (!found)
                    {
                        list.Insert(0, course);
                        stack.Pop();
                    }
                }
            }

            return list.ToArray();
        }

        // https://leetcode.com/problems/pacific-atlantic-water-flow/
        public IList<IList<int>> PacificAtlantic(int[][] heights)
        {
            int m = heights.Length, n = heights[0].Length;
            int[][] dirs = [[-1, 0], [0, 1], [1, 0], [0, -1]]; // up, right, down, left
            var list = new List<IList<int>>();

            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    var s = new Stack<int[]>();
                    var visited = new Dictionary<rNode, int>();
                    s.Push([i, j]);
                    bool canFlowPacific = false, canFlowAtlantic = false;
                    if (i == 0 || j == 0) canFlowPacific = true;
                    if (i == m - 1 || j == n - 1) canFlowAtlantic = true;

                    while (s.Count() > 0)
                    {
                        var cur = s.Pop();
                        foreach (var dir in dirs)
                        {
                            if (cur[0] + dir[0] < 0 || cur[0] + dir[0] > m - 1
                            || cur[1] + dir[1] < 0 || cur[1] + dir[1] > n - 1)
                                continue;

                            if (heights[cur[0]][cur[1]] >= heights[cur[0] + dir[0]][cur[1] + dir[1]]
                                && !visited.ContainsKey(new rNode(cur[0] + dir[0], cur[1] + dir[1])))
                            {
                                s.Push([cur[0] + dir[0], cur[1] + dir[1]]);
                                visited.Add(new rNode(cur[0] + dir[0], cur[1] + dir[1]), 0);
                                if (cur[0] + dir[0] == 0 || cur[1] + dir[1] == 0) canFlowPacific = true;
                                if (cur[0] + dir[0] == m - 1 || cur[1] + dir[1] == n - 1) canFlowAtlantic = true;
                            }
                        }

                        if (canFlowPacific && canFlowAtlantic) break;
                    }

                    if (canFlowPacific && canFlowAtlantic) list.Add(new List<int> { i, j });
                }
            }

            return list;
        }

        // https://leetcode.com/problems/clone-graph/
        public Node CloneGraph(Node node)
        {
            if (node == null) return null;

            var visited = new bool[101];
            var list = new Node[101];
            var q1 = new Queue<Node>();
            var q2 = new Queue<Node>();
            var newNode = new Node(1);
            list[1] = newNode;
            q1.Enqueue(node);
            q2.Enqueue(newNode);
            visited[1] = true;

            while (q1.Count() > 0)
            {
                Node c1 = q1.Dequeue(), c2 = q2.Dequeue();
                for (int i = 0; i < c1.neighbors.Count; i++)
                {
                    if (visited[c1.neighbors[i].val]) c2.neighbors.Add(list[c1.neighbors[i].val]);
                    else
                    {
                        c2.neighbors.Add(new Node(c1.neighbors[i].val));
                        list[c2.neighbors[i].val] = c2.neighbors[i];

                        q1.Enqueue(c1.neighbors[i]);
                        q2.Enqueue(c2.neighbors[i]);
                        visited[c2.neighbors[i].val] = true;
                    }
                }
            }

            return newNode;
        }

        // https://leetcode.com/problems/rotting-oranges/submissions/1670945123/
        // todo: learn multisource bfs
        public int OrangesRotting(int[][] grid)
        {
            int[][] dirs = [[-1, 0], [0, 1], [1, 0], [0, -1]]; // up, right, down, left
            int m = grid.Length, n = grid[0].Length;
            var dict = new Dictionary<rNode, int>();
            var visited = new Dictionary<rNode, bool>();
            var visitedGen = new Dictionary<rNode, bool>();
            var distances = new Dictionary<rNode, int>();

            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    if (grid[i][j] == 1 && (!visitedGen.ContainsKey(new rNode(i, j)) || !visitedGen[new rNode(i, j)]))
                        dict.Add(new rNode(i, j), 0);

                    if (grid[i][j] == 2)
                    {
                        visited.Clear();
                        grid[i][j] = 3;
                        var q = new Queue<int[]>();
                        q.Enqueue([i, j, 0]);

                        while (q.Count > 0)
                        {
                            var cur = q.Dequeue();
                            foreach (var dir in dirs)
                            {
                                if (cur[0] + dir[0] < 0 || cur[0] + dir[0] > m - 1
                                || cur[1] + dir[1] < 0 || cur[1] + dir[1] > n - 1)
                                    continue;

                                if (grid[cur[0] + dir[0]][cur[1] + dir[1]] == 1)
                                {

                                    if (!visited.ContainsKey(new rNode
                                    (cur[0] + dir[0], cur[1] + dir[1])))
                                        visited.Add(new rNode(cur[0] + dir[0], cur[1] + dir[1]), false);
                                    if (!visitedGen.ContainsKey(new rNode
                                    (cur[0] + dir[0], cur[1] + dir[1])))
                                        visitedGen.Add(new rNode(cur[0] + dir[0], cur[1] + dir[1]), false);
                                    if (!distances.ContainsKey(new rNode
                                        (cur[0] + dir[0], cur[1] + dir[1])))
                                        distances.Add(new rNode(cur[0] + dir[0], cur[1] + dir[1]), int.MaxValue);
                                    if (visited[new rNode(cur[0] + dir[0], cur[1] + dir[1])])
                                        continue;
                                    var key = new rNode(cur[0] + dir[0], cur[1] + dir[1]);
                                    if (dict.ContainsKey(key)) dict.Remove(key);
                                    // grid[cur[0] + dir[0]][cur[1] + dir[1]] = 3;
                                    q.Enqueue([cur[0] + dir[0], cur[1] + dir[1], cur[2] + 1]);
                                    visited[new rNode(cur[0] + dir[0], cur[1] + dir[1])] = true;
                                    var x = new rNode(cur[0] + dir[0], cur[1] + dir[1]);
                                    visitedGen[new rNode(cur[0] + dir[0], cur[1] + dir[1])] = true;
                                    distances[new rNode(cur[0] + dir[0], cur[1] + dir[1])] =
                                        Math.Min(distances[new rNode(cur[0] + dir[0], cur[1] + dir[1])], cur[2] + 1);
                                }
                            }
                        }
                    }
                }
            }

            foreach(var pair in dict) if (visitedGen.ContainsKey(pair.Key) && visitedGen[pair.Key]) dict.Remove(pair.Key);

            return dict.Count() > 0 ? -1 : distances.Count > 0 ? distances.Values.Max() : 0;
        }

        // https://leetcode.com/problems/max-area-of-island/description/
        public int MaxAreaOfIsland(int[][] grid)
        {
            int[][] dirs = [[-1, 0], [0, 1], [1, 0], [0, -1]]; // up, right, down, left
            int m = grid.Length, n = grid[0].Length, max = 0;

            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    if (grid[i][j] == 1)
                    {
                        int count = 0;
                        var stack = new Stack<int[]>();
                        grid[i][j] = -1;
                        stack.Push([i, j]);
                        while (stack.Count > 0)
                        {
                            count++;
                            var cur = stack.Pop();
                            foreach (var dir in dirs)
                            {
                                if (cur[0] + dir[0] < 0 || cur[0] + dir[0] > m - 1 || cur[1] + dir[1] < 0 || cur[1] + dir[1] > n - 1) continue;
                                if (grid[cur[0] + dir[0]][cur[1] + dir[1]] == 1)
                                {
                                    grid[cur[0] + dir[0]][cur[1] + dir[1]] = -1;
                                    stack.Push([cur[0] + dir[0], cur[1] + dir[1]]);

                                }
                            }
                        }

                        if (count > max) max = count;
                    }
                }
            }

            return max;
        }

        // https://leetcode.com/problems/number-of-islands/
        public int NumIslands(char[][] grid)
        {
            int[][] dirs = [[-1, 0], [0, 1], [1, 0], [0, -1]]; // up, right, down, left
            int m = grid.Length, n = grid[0].Length, count = 0;

            for (int i = 0; i < grid.Length; i++)
            {
                for (int j = 0; j < grid[i].Length; j++)
                {
                    if (grid[i][j] == '1')
                    {
                        count++;
                        var stack = new Stack<int[]>();
                        stack.Push([i,j]);
                        while(stack.Count > 0)
                        {
                            var cur = stack.Pop();
                            grid[cur[0]][cur[1]] = '-';
                            foreach (var dir in dirs)
                            {
                                if (cur[0] + dir[0] < 0 || cur[0] + dir[0] > m-1 || cur[1] + dir[1] < 0 || cur[1] + dir[1] > n-1) continue;
                                if (grid[cur[0] + dir[0]][cur[1] + dir[1]] == '1') stack.Push([cur[0] + dir[0], cur[1] + dir[1]]);
                            }                            
                        }
                    }
                }
            }

            return count;
        }

        public int LastStoneWeight(int[] stones)
        {
            var pq = new PriorityQueue<int[], int>();
            var list = stones.ToList();

            while (list.Count > 1)
            {
                BuildPq(pq, list);
                int[] i = pq.Dequeue(), j = pq.Dequeue();
                if (i[0] < j[0])
                {
                    list[j[1]] -= i[0];
                    list.RemoveAt(i[1]);
                }
                else
                {
                    list.RemoveAt(i[1]);
                    list.RemoveAt(j[1]-1);
                }
            }

            Console.WriteLine(string.Join(",", list));
            return list.FirstOrDefault();
        }

        public void BuildPq(PriorityQueue<int[], int> pq, List<int> stones)
        {
            pq.Clear();
            int i = 0, c = 0;
            while (c < 2 && i < stones.Count)
            {
                pq.Enqueue([stones[i], i], stones[i]);
                c++;
                i++;
            }
            for (int j = i; j < stones.Count; j++)
            {
                if (stones[j] > pq.Peek()[0])
                {
                    pq.Dequeue();
                    pq.Enqueue([stones[j], j], stones[j]);
                }
            }
        }
        public bool IsBalanced(TreeNode root)
        {
            bool balanced = true;
            DFS(root, 0, ref balanced);
            return balanced;
        }

        public int DFS(TreeNode root, int count, ref bool balanced)
        {
            if (root == null) return count;
            var left = DFS(root.left, count + 1, ref balanced);
            var right = DFS(root.right, count + 1, ref balanced);

            if (left != right) balanced = false;
            return count;
        }

        public bool IsPalindrome(string s)
        {
            var arr = s.ToCharArray();
            var arr2 = new List<char>();

            for (int i = 0; i < arr.Length; i++)
            {
                if ((arr[i] >= 'a' && arr[i] <= 'z') || (arr[i] >= 'A' && arr[i] <= 'Z') || (arr[i] >= '0' && arr[i] <= '9')) arr2.Add(Char.ToLower(arr[i]));
            }

            for (int i = 0, n = arr2.Count, j = n - 1; i < j; i++, j--)
            {
                if (arr2[i] != arr2[j]) return false;
            }

            return true;
        }

        public List<int> authEvents(List<List<string>> events)
        {
            // setPassword-000A
            // authorize-108738450
            // authorize-108738449
            // authorize-244736787

            int passwordHash = 0;
            var alts = new List<int>();
            var list = new List<int>();

            foreach (var e in events)
            {
                if (e[0] == "setPassword")
                {
                    passwordHash = hash(e[1]);
                    alts.Clear();
                    //Console.Write(hash(e[1] + 'z'));
                    for (int i = 1; i < 127; i++)
                    {
                        alts.Add(hash(e[1] + (char)i));
                    }
                }
                else
                {
                    int p = int.Parse(e[1]);
                    if (p == passwordHash) list.Add(1);
                    else if (alts.Contains(p)) list.Add(1);
                    else list.Add(0);
                }
            }

            // Console.Write(alts.Last());
            // Console.Write(string.Join(",",alts));
            return list;
        }

        public int hash(string s)
        {
            int k = 0;
            for (int i = 0, j = s.Length - 1; j >= 0; i++, j--)
            {
                k += f(s[i]) * (int)Math.Pow(131, j);
            }
            return k;
        }

        public int f(char c)
        {
            return (int)c;
        }

        // https://leetcode.com/problems/container-with-most-water/description/
        public int MaxArea(int[] height)
        {
            int maxArea = 0, i = 0, j = height.Length - 1;

            while (i < j)
            {
                maxArea = Math.Max(maxArea, (j - i) * Math.Min(height[j], height[i]));
                if (height[i] < height[j]) i++;
                else j--;
            }

            return maxArea;
        }

        // https://leetcode.com/problems/jump-game-ii/
        public int Jump(int[] nums)
        {
            // F(i) is the minimum number of jumps to reach the ith index
            // F(0) = 0, F(1) = nums[0] > 0 ? F(0) + 1 : 0

            int[] dp = new int[nums.Length];
            for (int i = 0; i < nums.Length; i++)
            {
                int c = 0;
                for (int j = i + 1; j < nums.Length && c < nums[i]; j++)
                {
                    if (dp[j] > 0) dp[j] = Math.Min(dp[j], dp[i] + 1);
                    else dp[j] = dp[i] + 1;
                    c++;
                }
            }

            return dp[dp.Length - 1];
        }

        // https://leetcode.com/problems/rotate-image/description/
        public void Rotate90(int[][] m)
        {
            // rotate 90 = transpose + reverse row
            // 180 = reverse row + reverse column
            // 270 = transpose + reverse col

            // transpose
            for (int i = 0; i < m.Length; i++)
            {
                for (int j = i + 1; j < m[i].Length; j++)
                {
                    int temp = m[i][j];
                    m[i][j] = m[j][i];
                    m[j][i] = temp;
                }
            }

            // reverse
            for (int i = 0; i < m.Length; i++)
            {
                for (int j = 0, k = m[i].Length - 1; j < k; j++, k--)
                {
                    int temp = m[i][j];
                    m[i][j] = m[i][k];
                    m[i][k] = temp;
                }
            }
        }

        public int KthSmallest(int[] arr, int k)
        {
            // Code Here
            for (int i = arr.Length / 2 - 1; i >= 0; i--)
            {
                Heapify(arr, i);
            }

            int c = 0;
            for (int i = 0; i < k; i++)
            {
                c = Dequeue(arr);
            }

            return c;
        }

        public void Heapify(int[] arr, int i)
        {
            int smallest = i;
            int l = i * 2 + 1, r = i * 2 + 2;

            if (arr[l] < arr[smallest]) smallest = l;
            if (r < arr.Length && arr[r] < arr[smallest]) smallest = r;

            if (smallest != i)
            {
                int temp = arr[i];
                arr[i] = arr[smallest];
                arr[smallest] = temp;

                Heapify(arr, i);
            }
        }

        public int Dequeue(int[] arr)
        {
            int temp = arr[0];
            arr[0] = arr[arr.Length - 1];
            arr[arr.Length - 1] = int.MaxValue;
            for (int i = arr.Length / 2 - 1; i >= 0; i--)
            {
                Heapify(arr, i);
            }
            return temp;
        }

        public int LengthOfLongestSubstring(string s)
        {
            if (string.IsNullOrEmpty(s)) return 0;

            int max = int.MinValue;
            var set = new HashSet<char>();
            for (int i = 0; i < s.Length; i++)
            {
                var sub = s.Substring(i);
                for (int j = 0; j < sub.Length - 1; j++)
                {
                    int size = set.Count();
                    set.Add(sub[j]);
                    if (set.Count == size)
                    {
                        max = Math.Max(max, size);
                        set.Clear();
                        break;
                    }
                }
                max = Math.Max(max, set.Count());
            }

            return max;
        }

        // https://leetcode.com/problems/count-and-say/description/
        public string CountAndSay(int n)
        {
            string prev = "1";
            var sb = new StringBuilder();

            for (int i = 1; i < n; i++)
            {
                int count = 1;
                for (int j = 1; j < prev.Length; j++)
                {
                    if (prev[j] != prev[j - 1])
                    {
                        sb.Append($"{count}{prev[j-1]}");
                        count = 1;
                    }
                    else count++;

                    if (j == prev.Length - 1)
                    {
                        sb.Append($"{count}{prev[j]}");
                    }
                }

                if (i == 1) sb.Append($"{count}{prev[0]}");
                prev = sb.ToString();
                sb.Clear();
            }

            return prev;
        }

        // https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/description/
        public int[] SearchRange(int[] nums, int target)
        {
            return binarySearch(nums, target, 0, nums.Length - 1, [-1, -1]);
        }

        public int[] binarySearch(int[] nums, int target, int start, int end, int[] f)
        {
            if (start > end) 
                return f;

            int mid = (start + end) / 2;

            if (nums[mid] == target)
            {
                f[1] = Math.Max(mid, f[1]);
                f[0] = f[0] != -1 ? Math.Min(mid, f[0]) : mid;
            }

            binarySearch(nums, target, start, mid - 1, f);
            binarySearch(nums, target, mid + 1, end, f);

            return f;
        }

        public int Search(int[] nums, int target)
        {
            return binarySearch(nums, target, 0, nums.Length - 1);
        }

        public int binarySearch(int[] nums, int target, int start, int end)
        {
            if (start > end) return -1;

            int mid = (start + end) / 2;
            if (nums[mid] == target) return mid;

            if (nums[start] > target && nums[mid] > target) return binarySearch(nums, target, mid + 1, end);
            else return binarySearch(nums, target, start, mid - 1);

            // if(nums[mid+1] > nums[mid] && target > nums[mid]) return binarySearch(nums, target, mid+1, end);
            // if(nums[start] > target && nums[mid] > target) return binarySearch(nums, target, mid+1, end);
            // else if(nums[end] < target && nums[mid] < target) return binarySearch(nums, target, start, mid-1);
            // else if(nums[mid] > target) return binarySearch(nums, target, start, mid-1);
            // else return binarySearch(nums, target, mid+1, end);
        }

        public int Divide(int dividend, int divisor)
        {
            long q = 0, t1 = dividend, t2 = divisor;

            while (Math.Abs(t1) >= Math.Abs(t2))
            {
                t1 = Math.Abs(t1) - Math.Abs(t2);
                q++;
            }

            return ((divisor > 0 && dividend > 0) || (divisor < 0 && dividend < 0)) ? (int)q : (int)-q;
        }

        // https://leetcode.com/problems/swap-nodes-in-pairs/description/
        public ListNode SwapPairs(ListNode head)
        {
            ListNode prev = head, cur = head?.next, past = null;

            while (cur != null && prev != null)
            {
                prev.next = cur.next;
                cur.next = prev;
                if (prev.Equals(head)) head = cur;

                if (past != null) past.next = cur;
                past = prev;
                cur = prev.next?.next;
                prev = prev.next;
            }

            return head;
        }
        public int ThreeSumClosest(int[] nums, int target)
        {
            int min = int.MaxValue, sum = 0;

            for (int i = 0; i < nums.Length; i++)
            {
                for (int j = 0; j < nums.Length; j++)
                {
                    for (int k = 0; k < nums.Length; k++)
                    {
                        if (i != j && j != k && i != k && (Math.Abs((nums[i] + nums[j] + nums[k]) - target) < min))
                        {
                            sum = nums[i] + nums[j] + nums[k];
                            min = (nums[i] + nums[j] + nums[k]) - target;
                            Console.WriteLine($"{sum}-{min}");
                        }
                    }
                }
            }

            return sum;
        }

        public async Task RunGrpc()
        {
            using var channel = GrpcChannel.ForAddress("https://localhost:7241");
            var client = new Greeter.GreeterClient(channel);
            var reply = await client.SayHelloAsync(
                new HelloRequest { Name = "GreeterClient" });
            Console.WriteLine("Greeting: " + reply.Message);
            Console.WriteLine("Press any key to exit...");
            Console.ReadKey();
        }

        public IList<IList<int>> ThreeSum(int[] nums)
        {
            var list = new List<IList<int>>();
            var dict = new Dictionary<string, int>();

            for (int i = 0; i < nums.Length && nums[i] > 0; i++)
            {
                for (int j = 0; j < nums.Length && nums[j] <= 0; j++)
                {
                    for (int k = 0; k < nums.Length && nums[k] <= 0; k++)
                    {
                        if (i != j && j != k && i != k && (nums[i] + nums[j] + nums[k] == 0))
                        {
                            var l = new List<int> { nums[i], nums[j], nums[k] };
                            l.Sort();
                            if (dict.ContainsKey(string.Join("", l))) continue;
                            dict.Add(string.Join("", l), 0);
                            list.Add(l);
                        }
                    }
                }
            }

            return list;
        }

        // https://leetcode.com/problems/add-two-numbers/description/
        public ListNode AddTwoNumbers(ListNode l1, ListNode l2)
        {
            ListNode cur1 = l1.next, cur2 = l2.next, res = new ListNode(l1.val + l2.val), cur3 = res;
            bool rem = false;
            if (res.val > 9)
            {
                res.val -= 10;
                rem = true;
            }

            while (cur1 != null && cur2 != null)
            {
                if (rem)
                {
                    cur1.val++;
                    rem = false;
                }

                int sum = cur1.val + cur2.val;
                if (sum > 9)
                {
                    sum -= 10;
                    rem = true;
                }
                cur3.next = new ListNode(sum);
                cur1 = cur1.next;
                cur2 = cur2.next;
                cur3 = cur3.next;
            }

            while (cur1 != null)
            {
                if (rem)
                {
                    cur1.val++;
                    rem = false;
                }

                if (cur1.val > 9)
                {
                    cur1.val -= 10;
                    rem = true;
                }

                cur3.next = new ListNode(cur1.val);
                cur1 = cur1.next;
                cur3 = cur3.next;
            }

            while (cur2 != null)
            {
                if (rem)
                {
                    cur2.val++;
                    rem = false;
                }

                if (cur2.val > 9)
                {
                    cur2.val -= 10;
                    rem = true;
                }

                cur3.next = new ListNode(cur2.val);
                cur2 = cur2.next;
                cur3 = cur3.next;
            }

            if (rem) cur3.next = new ListNode(1);

            return res;
        }

        // https://leetcode.com/problems/integer-to-roman/description/
        public string IntToRoman(int num)
        {
            var s = new StringBuilder();
            string sum = num.ToString();

            for (int i = 0; i < sum.Length; i++)
            {
                if (sum[i] == '0') continue;

                if (sum[i] == '4' || sum[i] == '9')
                {
                    int l = sum.Substring(i).Length;
                    switch (l)
                    {
                        case 3:
                            if (sum[i] == '4') s.Append("CD");
                            else s.Append("CM");
                            break;
                        case 2:
                            if (sum[i] == '4') s.Append("XL");
                            else s.Append("XC");
                            break;
                        case 1:
                            if (sum[i] == '4') s.Append("IV");
                            else s.Append("IX");
                            break;
                        default:
                            break;
                    }
                }
                else
                {
                    int n = int.Parse(sum.Substring(i));
                    if (n >= 1000)
                    {
                        for (int q = 0; q < n / 1000; q++) s.Append("M");
                    }
                    else if (n >= 500)
                    {
                        s.Append("D");
                        for (int q = 0; q < sum[i] - '5'; q++) s.Append("C");
                    }
                    else if (n >= 100)
                    {
                        for (int q = 0; q < sum[i] - '0'; q++) s.Append("C");
                    }
                    else if (n >= 50)
                    {
                        s.Append("L");
                        for (int q = 0; q < sum[i] - '5'; q++) s.Append("X");
                    }
                    else if (n >= 10)
                    {
                        s.Append("X");
                        for (int q = 0; q < sum[i] - '1'; q++) s.Append("X");
                    }
                    else if (n >= 5)
                    {
                        s.Append("V");
                        for (int q = 0; q < sum[i] - '5'; q++) s.Append("I");
                    }
                    else
                    {
                        for (int q = 0; q < sum[i] - '0'; q++) s.Append("I");
                    }
                }
            }

            return s.ToString();
        }

        // https://leetcode.com/problems/maximum-product-of-word-lengths/
        public int MaxProduct(string[] words)
        {
            int max = 0;
            var arr = new string[words.Length];

            for (int i = 0; i < words.Length; i++)
            {
                arr[i] = new string(words[i].Distinct().ToArray());
            }

            for (int i = 0; i < arr.Length; i++)
            {
                for (int j = i + 1; j < arr.Length; j++)
                {
                    bool match = true;
                    for (int k = 97; k < 123; k++)
                    {
                        if (arr[i].Contains((char)k) && arr[j].Contains((char)k))
                        {
                            match = false;
                            break;
                        }
                    }
                    if (match && words[i].Length * words[j].Length > max) max = words[i].Length * words[j].Length;
                }
            }
            return max;
        }


        // https://leetcode.com/problems/sum-root-to-leaf-numbers/
        public int SumNumbers(TreeNode root)
        {
            int total = 0;
            dfs(root, "", ref total);
            return total;
        }

        public void dfs(TreeNode root, string sum, ref int total)
        {
            if (root == null) return;
            string sum2 = sum + root.val;
            if (root.left == null && root.right == null) total += int.Parse(sum2);
            dfs(root.left, sum2, ref total);
            dfs(root.right, sum2, ref total);
        }

        // https://leetcode.com/problems/find-peak-element/
        public int FindPeakElement(int[] nums)
        {
            int max = 0;
            return Bin(nums, 0, nums.Length - 1, ref max);
        }

        public int Bin(int[] nums, int start, int end, ref int max)
        {
            if (end == start) return max;
            if (nums[end] > nums[end - 1])
            {
                if (end == nums.Length - 1 || nums[end] > nums[end + 1]) max = end;
            }
            if (nums[start] > nums[start + 1])
            {
                if (start == 0 || nums[start] > nums[start - 1]) max = start;
            }

            int mid = (start + end) / 2;
            if (nums[mid] > nums[mid + 1])
                return Bin(nums, start, mid, ref max);
            return Bin(nums, mid + 1, end, ref max);
        }

        // https://leetcode.com/problems/find-largest-value-in-each-tree-row/description/
        public IList<int> LargestValues(TreeNode root)
        {
            var list = new List<int>();
            if (root == null) return list;
            Queue<(TreeNode node, int level)> q = new Queue<(TreeNode, int)>();
            q.Enqueue((root, 0));
            list.Add(root.val);

            while (q.Count > 0)
            {
                var cur = q.Dequeue();
                if (cur.node.val > list[cur.level]) list[cur.level] = cur.node.val;

                if ((cur.node.left != null || cur.node.right != null) && list.Count == cur.level + 1) list.Add(int.MinValue);
                if (cur.node.left != null)
                {
                    q.Enqueue((cur.node.left, cur.level + 1));
                }
                if (cur.node.right != null)
                {
                    q.Enqueue((cur.node.right, cur.level + 1));
                }
            }

            return list;
        }

        // https://leetcode.com/problems/maximum-gap/
        public int MaximumGap(int[] nums)
        {
            if (nums.Length < 2) return 0;
            int max = int.MinValue, min = int.MaxValue, maxGap = 0;
            for (int i = 0; i < nums.Length; i++)
            {
                if (nums[i] < min) min = nums[i];
                if (nums[i] > max) max = nums[i];
            }

            if (max == min) return 0;

            int[] bucket = new int[(max - min) + 1];
            for (int i = 0; i < nums.Length; i++)
            {
                bucket[nums[i] - min] = -1;
            }

            int? last = bucket[min - min] == -1 ? 0 : null;
            for (int i = 1; i < bucket.Length; i++)
            {
                if (!last.HasValue && bucket[i] == -1)
                {
                    last = i;
                    continue;
                }

                if (bucket[i] == -1)
                {
                    if (i - last.Value > maxGap) maxGap = i - last.Value;
                    last = i;
                }
            }

            return maxGap;
        }

        public int CoinChange(int[] coins, int amount)
        {
            int num = 0;
            Array.Sort(coins);
            for (int i = coins.Length-1; i >= 0; i--)
            {
                if(amount / coins[i] > 1)
                {
                    num += amount / coins[i];
                    amount %= coins[i];
                }
            }

            return amount == 0 ? num : -1;
        }

        public int maxCreds(int[] wh, int d1, int d2, int skips)
        {
            int creds = 0;

            for (int i = 0; i < wh.Length; i++)
            {
                bool d1Turn = true;

                while (wh[i] > 0)
                {
                    if (wh[i] <= d2 && !d1Turn && skips > 0)
                    {
                        d1Turn = true;
                        skips--;
                    }

                    if(d1Turn)
                    {
                        wh[i] -= d1;
                    }
                    else
                    {
                        wh[i] -= d2;
                    }

                    if (wh[i] <= 0 && d1Turn)
                    {
                        creds++;
                    }
                    d1Turn = !d1Turn;
                }
            }

            return creds;
        }

        // https://leetcode.com/problems/remove-duplicates-from-sorted-list-ii/
        public ListNode DeleteDuplicates(ListNode head)
        {
            int rand = new Random().Next(0,10);
            ListNode cur = head, prev = null;
            while (cur != null)
            {
                bool isDup = false;
                while (cur.next != null && cur.val == cur.next.val)
                {
                    isDup = true;
                    cur.next = cur.next.next;
                }

                if (isDup)
                {
                    if (prev == null)
                    {
                        head = cur.next;
                        cur = head;
                        continue;
                    }
                    else if (cur.next == null)
                    {
                        prev.next = null;
                        return head;
                    }
                    else
                    {
                        prev.next = cur.next;
                        cur = cur.next;
                        continue;
                    }
                }

                prev = cur;
                cur = cur.next;
            }

            return head;
        }

        // https://leetcode.com/problems/letter-combinations-of-a-phone-number/
        public IList<string> LetterCombinations(string digits)
        {
            var list = new List<string>();
            if (string.IsNullOrEmpty(digits)) return list;
            var dict = new Dictionary<int, string>{
            {2,"abc"},
            {3,"def"},
            {4,"ghi"},
            {5,"jkl"},
            {6,"mno"},
            {7,"pqrs"},
            {8,"tuv"},
            {9,"wxyz"}
        };

            Combine(list, "", digits, dict);

            return list;
        }

        public void Combine(List<string> list, string s, string digits, Dictionary<int, string> dict)
        {
            if (digits.Length == 0)
            {
                list.Add(s);
                return;
            }
            int num = digits[0] - '0';

            foreach (char c in dict[num])
            {
                Combine(list, s + c, digits.Length > 1 ? digits.Substring(1) : "", dict);
            }
        }

        // https://leetcode.com/problems/combinations/description/
        public IList<IList<int>> Combine(int n, int k)
        {
            var list = new List<IList<int>>();
            for (int i = 1; i <= n; i++)
            {
                Combine(list, new List<int> { i }, n, i, k);
            }
            return list;
        }

        public void Combine(List<IList<int>> resp, List<int> list, int n, int num, int k)
        {
            if (num > n) return;
            if (list.Count == k)
            {
                resp.Add(new List<int>(list));
                list.RemoveAt(list.Count-1);
                return;
            }

            int c = num + 1;
            while (c <= n)
            {
                list.Add(c);
                Combine(resp, list, n, c, k);
                c++;
            }
            list.RemoveAt(list.Count - 1);
        }

        public static int FindMinDifference(IList<string> t)
        {
            int min = int.MaxValue;
            var list = new List<int>();

            foreach (var s in t)
            {
                var arr = s.Split(':');
                list.Add(int.Parse(arr[0]) * 60 + int.Parse(arr[1]));
            }
            list.Sort();

            for(int i=1; i<list.Count; i++)
            {
                int diff = list[i] - list[i - 1];
                if (diff == 0) return 0;
                if (diff >= 0 && diff < min) min = diff;
            }

            int wrapDiff = 1440 - list[list.Count - 1] + list[0];
            if (wrapDiff >= 0 && wrapDiff < min) min = wrapDiff;

            return min;
        }

        public int[] PlusOne(int[] digits)
        {
            int i = digits.Length - 1;
            while (i >= 0 && digits[i] == 9)
            {
                digits[i] = 0;
                i--;
            }
            if (i > -1)
            {
                digits[i]++;
                return digits;
            }
            digits[0] = 1;
            var list = digits.ToList();
            list.Add(0);
            return list.ToArray();
        }

        // https://leetcode.com/problems/equal-sum-grid-partition-i/description/
        public static bool CanPartitionGrid(int[][] grid)
        {
            int m = grid.Length, n = grid[0].Length;
            int[] cols = new int[m];
            int[] rows = new int[n];

            for (int k = 0; k < grid.Length; k++)
            {
                for (int l = 0; l < grid[k].Length; l++)
                {
                    cols[k] += grid[k][l];
                    rows[l] += grid[k][l];
                }
            }

            long i = 0, j = m - 1, left = 0, right = 0;
            while (i <= j)
            {
                if (left == right)
                {
                    if (i == j && cols.Length > 1) return false;
                    left += cols[i];
                    right += cols[j];
                    i++;
                    j--;
                    continue;
                }
                else if (left > right)
                {
                    right += cols[j];
                    j--;
                    continue;
                }
                else
                {
                    left += cols[i];
                    i++;
                    continue;
                }
            }

            i = 0; j = n - 1;
            long up = 0, down = 0;
            while (i <= j)
            {
                if (up == down)
                {
                    if (i == j && rows.Length > 1) return false;
                    up += rows[i];
                    down += rows[j];
                    i++;
                    j--;
                    continue;
                }
                else if (up > down)
                {
                    down += rows[j];
                    j--;
                    continue;
                }
                else
                {
                    up += rows[i];
                    i++;
                    continue;
                }
            }

            return (cols.Length > 1 && left == right) || (rows.Length > 1 && up == down);
        }

        // https://leetcode.com/problems/repeated-string-match/
        public int RepeatedStringMatch(string a, string b)
        {
            if (string.IsNullOrEmpty(b)) return 0;

            var sb = new StringBuilder();
            sb.Append(a);
            int c = 1;

            while (sb.Length < b.Length)
            {
                sb.Append(a);
                c++;
            }

            if (sb.ToString().Contains(b)) return c;
            sb.Append(a);
            c++;
            if (sb.ToString().Contains(b)) return c;
            return -1;
        }

        // https://leetcode.com/problems/find-minimum-time-to-reach-last-room-i/description/
        public int MinTimeToReach(int[][] m)
        {
            int mY = m.Length, mX = m[0].Length;
            bool[,] visited = new bool[mY, mX];

            var pq = new PriorityQueue<(int i, int j, int t), int>();
            pq.Enqueue((0, 0, 0), 0);
            // top, right, down, left
            int[][] dirs = new int[][] { new int[] { -1, 0 }, new int[] { 0, 1 }, new int[] { 1, 0 }, new int[] { 0, -1 } };

            while (pq.Count > 0)
            {
                var u = pq.Dequeue();

                if (visited[u.i, u.j]) continue;
                visited[u.i, u.j] = true;
                if ((u.i, u.j) == (mY - 1, mX - 1)) return u.t;

                foreach (var dir in dirs)
                {
                    int y = u.i + dir[0];
                    int x = u.j + dir[1];
                    if (y < 0 || y >= mY || x < 0 || x >= mX) continue;

                    (int i, int j) neighbor = (y, x);

                    if (!visited[neighbor.i, neighbor.j])
                    {
                        int alt = Math.Max(m[neighbor.i][neighbor.j], u.t) + 1;

                        pq.Enqueue((neighbor.i, neighbor.j, alt), alt);
                    }
                }
            }

            return -1;
        }

        // https://leetcode.com/problems/minimum-deletions-for-at-most-k-distinct-characters/description/
        public static int MinDeletion(string s, int k)
        {
            var dict = new Dictionary<char, int>();
            foreach (char c in s)
            {
                if (!dict.ContainsKey(c)) dict.Add(c, 1);
                else dict[c]++;
            }

            dict = dict.OrderBy(x => x.Value).ToDictionary();
            int count = 0;

            if (dict.Count == k) return 0;
            foreach (var item in dict)
            {
                count += item.Value;
                dict.Remove(item.Key);
                if (dict.Count == k) return count;
            }
            return 0;
        }

        public bool IsValidBST(TreeNode root)
        {
            var list = new List<int>();
            return DFS(root, list);
        }

        public bool DFS(TreeNode root, List<int> list)
        {
            if (root == null) return true;
            bool left = DFS(root.left, list);
            if (list.Count > 0 && root.val <= list[list.Count - 1]) return false;
            list.Add(root.val);
            bool right = DFS(root.right, list);
            return left && right;
        }

        public TreeNode ConvertBST(TreeNode root)
        {
            var list = new List<TreeNode>();
            DFS(root, 0);
            return root;
        }

        public int DFS(TreeNode root, int sum)
        {
            if (root == null) return sum;
            sum = DFS(root.right, sum);
            int temp = root.val;
            root.val += sum;
            sum += temp;
            sum = DFS(root.left, sum);
            return sum;
        }

        public static string MinRemoveToMakeValid(string s)
        {
            Stack<(char c, int index)> stack = new Stack<(char, int)>();
            var list = s.ToCharArray().ToList();

            for (int i = 0; i < list.Count; i++)
            {
                if (list[i] == '(') stack.Push((list[i], i));
                else if (list[i] == ')')
                {
                    if (stack.Count < 1)
                    {
                        list.RemoveAt(i);
                        i--;
                    }
                    else stack.Pop();
                }
            }

            while (stack.Count > 0)
            {
                var t = stack.Pop();
                list.RemoveAt(t.index);
            }

            return string.Join("", list);
        }

        public Node connect(Node root)
        {
            Queue<(Node node, int level)> q = new Queue<(Node, int)>();
            var dict = new Dictionary<int, List<Node>>();
            q.Enqueue((root,1));

            while (q.Count > 0)
            {
                var cur = q.Dequeue();
                int lev = cur.level + 1;
                if (cur.node.left == null) break;

                q.Enqueue((cur.node.left, lev));
                q.Enqueue((cur.node.right, lev));

                if (dict.ContainsKey(lev))
                {
                    dict[lev].Add(cur.node.left);
                    dict[lev].Add(cur.node.right);
                }
                else
                {
                    dict.Add(lev, new List<Node> { cur.node.left, cur.node.right });
                }
            }

            foreach (var item in dict)
            {
                for (int i = 0; i <= item.Value.Count - 1; i++)
                {
                    if (i + 1 < item.Value.Count) item.Value[i].next = item.Value[i + 1];
                }
            }

            return root;
        }

        public static IList<IList<string>> Partition(string s)
        {
            var list = new List<IList<string>>();

            var l2 = new List<string>();
            for (int i = 0; i < s.Length; i++)
            {
                for (int j = i; j < s.Length; j++)
                {
                    if (s[j] == s[i])
                    {
                        //check if s[0-i] ispal and s[i+1-end] ispal
                        string s1 = s.Substring(i, j + 1), s2 = s.Substring(j + 1);
                        if (isPal(s1) && isPal(s2))
                        {
                            var l3 = new List<string>();
                            if (!string.IsNullOrEmpty(s1)) l3.Add(s1);
                            if (!string.IsNullOrEmpty(s2)) l3.Add(s2);
                            list.Add(l3);
                            i = j;
                            break;
                        }
                    }
                }

            }

            list.Add(l2);
            return list;
        }

        public static bool isPal(string s)
        {
            for (int i = 0, j = s.Length - 1; i < j; i++, j--)
            {
                if (s[i] != s[j]) return false;
            }
            return true;
        }

        public static int CanCompleteCircuit(int[] gas, int[] cost)
        {
            int gaso = 0;
            for (int i = 0; i < gas.Length; i++)
            {
                bool canGo = false;
                if (gas[i] >= cost[i])
                {
                    canGo = true;
                    gaso = gas[i];
                    for (int j = i + 1; j < gas.Length; j++)
                    {
                        gaso += gas[j];
                        gaso -= cost[j - 1];
                        if (gaso < cost[j])
                        {
                            canGo = false;
                            break;
                        }
                    }
                    if (!canGo) continue;
                    for (int j = 0; j < i; j++)
                    {
                        gaso += gas[j];
                        int prev = j == 0 ? gas.Length - 1 : j - 1;
                        gaso -= cost[prev];
                        if (gaso < cost[j])
                        {
                            canGo = false;
                            break;
                        }
                    }
                    if (!canGo) continue;
                    canGo = false;
                    int prevI = i == 0 ? gas.Length - 1 : i - 1;
                    if (gaso >= cost[prevI]) canGo = true;
                }
                if (canGo) return i;
                gaso = 0;
            }

            return -1;
        }

        // https://leetcode.com/problems/maximum-repeating-substring
        public int MaxRepeating(string sequence, string word)
        {
            for (int i = 100; i > 0; i--)
            {
                var s = new StringBuilder().Insert(0, word, i).ToString();
                if (sequence.IndexOf(s) > -1) return i;
            }
            return 0;
        }

        // https://leetcode.com/problems/longest-unequal-adjacent-groups-subsequence-i
        public IList<string> GetLongestSubsequence(string[] words, int[] groups)
        {
            var list = new List<string> { words[0] };
            int last = groups[0];

            for (int i = 1; i < groups.Length; i++)
            {
                if (groups[i] != last)
                {
                    list.Add(words[i]);
                    last = groups[i];
                }
            }

            return list;
        }

        // https://www.hackerrank.com/challenges/encryption/problem?isFullScreen=true
        public static string encryption(string s)
        {
            int floor = (int)Math.Floor(Math.Pow(s.Length, 0.5)), ceil = (int)Math.Ceiling(Math.Pow(s.Length, 0.5));

            while (floor * ceil < s.Length) floor++;

            string[] enc = new string[ceil];
            int sIndex = 0;
            for (int i = 0; i < floor; i++)
            {
                for (int j = 0; j < ceil; j++)
                {
                    if (sIndex >= s.Length) break;
                    enc[j] += s[sIndex];
                    sIndex++;
                }
            }

            return string.Join(' ', enc);
        }

        // https://www.hackerrank.com/challenges/birthday-cake-candles/problem
        public static int birthdayCakeCandles(List<int> candles)
        {
            var dict = new Dictionary<int, int>();
            int max = candles[0];

            foreach (int num in candles)
            {
                if (dict.ContainsKey(num)) dict[num]++;
                else dict.Add(num, 1);
                if (num > max) max = num;
            }

            return dict[max];
        }

        // https://www.hackerrank.com/challenges/apple-and-orange/problem
        public static void countApplesAndOranges(int s, int t, int a, int b, List<int> apples, List<int> oranges)
        {
            int x = 0, y = 0;
            foreach (int num in apples)
            {
                if (a + num >= s && a + num <= t) x++;
            }
            foreach (int num in oranges)
            {
                if (b + num <= t && b + num >= s) y++;
            }

            Console.WriteLine(x);
            Console.WriteLine(y);
        }

        // https://www.hackerrank.com/challenges/the-time-in-words/problem
        public static string timeInWords(int h, int m)
        {
            var dict = new Dictionary<int, string>(){
              {1,"one"},
              {2,"two"},
              {3,"three"},
              {4,"four"},
              {5,"five"},
              {6,"six"},
              {7,"seven"},
              {8,"eight"},
              {9,"nine"},
              {10,"ten"},
              {11,"eleven"},
              {12,"twelve"},
              {13,"thirteen"},
              {14,"fourteen"},
              {15,"fifteen"},
              {16,"sixteen"},
              {17,"seventeen"},
              {18,"eighteen"},
              {19,"nineteen"}
            };

            if (m == 0) return $"{dict[h]} o' clock";
            if (m == 15) return $"quarter past {dict[h]}";
            if (m < 30)
            {
                if (m < 20) return $"{dict[m]} minutes past {dict[h]}";
                var s = m.ToString();
                return $"twenty {(m == 20 ? "" : dict[int.Parse(s[s.Length - 1].ToString())])} minutes past {dict[h]}";
            }
            if (m == 30) return $"half past {dict[h]}";
            if (m == 45) return $"quarter to {dict[h]}";

            // m > 30
            m = 60 - m;
            if (m < 20) return $"{dict[m]} minutes to {dict[h + 1]}";
            var str = m.ToString();
            return $"twenty {(m == 20 ? "" : dict[int.Parse(str[str.Length - 1].ToString())])} minutes to {dict[h + 1]}";
        }

        public static List<int> climbingLeaderboard(List<int> ranked, List<int> player)
        {
            var list = new List<int>();
            int count = 1, prev = ranked[0], i = player.Count - 1;

            foreach (int num in ranked)
            {
                if (num < prev)
                {
                    count++;
                }
                if (player[i] >= num)
                {
                    i--;
                    list.Insert(0, count);
                }
            }

            return list;
        }

        // https://leetcode.com/problems/path-sum-ii/description/?envType=problem-list-v2&envId=binary-tree
        public static IList<IList<int>> PathSum(TreeNode root, int targetSum)
        {
            var stack = new Stack<TreeNode>();
            var res = new List<IList<int>>();
            var visited = new Dictionary<TreeNode, bool>();

            if (root == null) return res;
            stack.Push(root);
            visited.Add(root, true);
            int sum = root.val;

            while (stack.Count > 0)
            {
                var cur = stack.Peek();
                if (cur.left != null && !visited.ContainsKey(cur.left))
                {
                    stack.Push(cur.left);
                    visited.Add(cur.left, true);
                    sum += cur.left.val;
                    continue;
                }
                if (cur.right != null && !visited.ContainsKey(cur.right))
                {
                    stack.Push(cur.right);
                    visited.Add(cur.right, true);
                    sum += cur.right.val;
                    continue;
                }

                if (cur.left == null && cur.right == null)
                {
                    if (sum == targetSum)
                    {
                        var list = stack.ToList();
                        var list2 = new List<int>();
                        for (int i = list.Count - 1; i >= 0; i--)
                        {
                            list2.Add(list[i].val);
                        }

                        res.Add(list2);
                    }
                }

                sum -= cur.val;
                stack.Pop();
            }

            return res;
        }

        // https://leetcode.com/problems/binary-tree-level-order-traversal/description/?envType=problem-list-v2&envId=binary-tree
        public static IList<IList<int>> LevelOrder(TreeNode root)
        {
            var res = new List<IList<int>>();
            if (root == null) return res;

            int level = 1;
            var list = new List<int>();
            var q = new Queue<(TreeNode n, int l)>();
            q.Enqueue((root, 1));

            while (q.Count > 0)
            {
                var cur = q.Dequeue();
                if (list.Count == 0) list.Add(cur.n.val);
                else
                {
                    if (cur.l != level)
                    {
                        res.Add(new List<int>(list));
                        list.Clear();
                        list.Add(cur.n.val);
                        level++;
                    }
                    else list.Add(cur.n.val);
                }

                if (cur.n.left != null) q.Enqueue((cur.n.left, cur.l + 1));
                if (cur.n.right != null) q.Enqueue((cur.n.right, cur.l + 1));
            }

            res.Add(new List<int>(list));
            return res;
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

        public TreeNode BuildTree(int?[] values)
        {
            if (values == null || values.Length == 0 || values[0] == null)
                return null;

            TreeNode root = new TreeNode(values[0].Value);
            Queue<TreeNode> queue = new Queue<TreeNode>();
            queue.Enqueue(root);

            int i = 1;
            while (i < values.Length)
            {
                TreeNode current = queue.Dequeue();

                // Left child
                if (i < values.Length && values[i] != null)
                {
                    current.left = new TreeNode(values[i].Value);
                    queue.Enqueue(current.left);
                }
                i++;

                // Right child
                if (i < values.Length && values[i] != null)
                {
                    current.right = new TreeNode(values[i].Value);
                    queue.Enqueue(current.right);
                }
                i++;
            }

            return root;
        }
    }
}
