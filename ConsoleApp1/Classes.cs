using System;
using System.Collections.Generic;

namespace ConsoleApp1
{
    public class ListNode {
        public int val;
        public ListNode next;
        public ListNode(int val=0, ListNode next=null) {
            this.val = val;
            this.next = next;
        }
    }

    public class TreeNode
    {
        public int val;
        public TreeNode left;
        public TreeNode right;
        public TreeNode(int val = 0, TreeNode left = null, TreeNode right = null)
        {
            this.left = left;
            this.right = right;
            this.val = val;
        }
    }

    public class Node
    {
        public int val;
        public Node left;
        public Node right;
        public Node next;
        public IList<Node> neighbors;

        public Node() { }

        public Node(int _val)
        {
            val = _val;
        }

        public Node(int _val, Node _left, Node _right, Node _next)
        {
            val = _val;
            left = _left;
            right = _right;
            next = _next;
        }
    };

    // https://leetcode.com/problems/time-based-key-value-store/description/
    public class TimeMap
    {
        Dictionary<string, List<(string val, int timestamp)>> map;
        public TimeMap()
        {
            map = new();
        }

        public void Set(string key, string value, int timestamp)
        {
            if (!map.ContainsKey(key)) map.Add(key, new List<(string val, int timestamp)> { (value, timestamp) });
            else map[key].Add((value, timestamp));
        }

        public string Get(string key, int timestamp)
        {
            if (!map.ContainsKey(key)) return "";
            return BinarySearch(map[key], timestamp, 0, map[key].Count - 1);
        }

        public string BinarySearch(List<(string val, int timestamp)> list, int timestamp, int start, int end)
        {
            if (start > end) return "";

            int mid = (start + end) / 2;
            if (list[mid].timestamp == timestamp) return list[mid].val;
            if (list[mid].timestamp < timestamp)
            {
                if (mid < list.Count - 1 && timestamp < list[mid + 1].timestamp) return list[mid].val;
                if (mid == list.Count - 1) return list[mid].val;

                return BinarySearch(list, timestamp, mid + 1, end);
            }
            return BinarySearch(list, timestamp, start, mid - 1);
        }
    }

    // https://leetcode.com/problems/detect-squares/
    public class DetectSquares
    {
        Dictionary<int, List<int[]>> mapX;
        Dictionary<int, List<int[]>> mapY;
        Dictionary<(int x, int y), int> map;

        public DetectSquares()
        {
            mapX = new();
            mapY = new();
            map = new();
        }

        public void Add(int[] point)
        {
            if (!mapX.ContainsKey(point[0])) mapX.Add(point[0], new List<int[]> { point });
            else mapX[point[0]].Add(point);

            if (!mapY.ContainsKey(point[1])) mapY.Add(point[1], new List<int[]> { point });
            else mapY[point[1]].Add(point);

            if (!map.ContainsKey((point[0], point[1]))) map.Add((point[0], point[1]), 1);
            else map[(point[0], point[1])]++;
        }

        public int Count(int[] point)
        {
            if (!mapX.ContainsKey(point[0]) || !mapY.ContainsKey(point[1])) return 0;
            int count = 0;
            var processed = new HashSet<(int, int)>();

            // check along x-axis
            // Console.WriteLine(mapY[point[1]].Count);
            foreach (var p in mapY[point[1]])
            {
                if (p[0] == point[0]) continue;
                if (!processed.Add((p[0], p[1]))) continue;
                processed.Add((p[0], p[1]));

                int diff = Math.Abs(point[0] - p[0]);
                int c = 0;

                // Check UP
                var p1 = (p[0], p[1] + diff);
                var p2 = (point[0], point[1] + diff);
                if (map.ContainsKey(p1) && map.ContainsKey(p2))
                {
                    count += map[(p[0], p[1])] * map[p1] * map[p2];
                }

                // Check DOWN
                p1 = (p[0], p[1] - diff);
                p2 = (point[0], point[1] - diff);
                if (map.ContainsKey(p1) && map.ContainsKey(p2))
                {
                    count += map[(p[0], p[1])] * map[p1] * map[p2];
                }
            }

            return count;
        }
    }

    public class WordDictionary
    {
        public TrieNode root;

        public WordDictionary()
        {
            root = new TrieNode();
        }

        public void AddWord(string word)
        {
            var cur = root;
            foreach (char c in word)
            {
                if (cur.nodes[c - 'a'] == null)
                {
                    cur.nodes[c - 'a'] = new TrieNode();
                }
                cur = cur.nodes[c - 'a'];
            }

            cur.isWordEnd = true;
        }

        public bool Search(string word)
        {
            return search(word, 0, root);
        }

        bool search(string word, int n, TrieNode cur)
        {
            if (n > word.Length) return true;
            for (int i = n; i < word.Length; i++)
            {
                if (word[i] == '.')
                {
                    for (int j = 0; j < cur.nodes.Length; j++)
                    {
                        if (cur.nodes[j] == null) continue;
                        bool res = search(word, i + 1, cur.nodes[j]);
                        if (res) return true;
                    }
                    return false;
                }
                else if (cur.nodes[word[i] - 'a'] == null) return false;
                else cur = cur.nodes[word[i] - 'a'];
            }

            return cur.isWordEnd;
        }
    }

    public class Trie
    {
        public TrieNode root;

        public Trie()
        {
            root = new TrieNode();
        }

        public void Insert(string word)
        {
            var cur = root;
            foreach (char c in word)
            {
                if (cur.nodes[c - 'a'] == null)
                {
                    cur.nodes[c - 'a'] = new TrieNode();
                }
                cur = cur.nodes[c - 'a'];
            }

            cur.isWordEnd = true;
        }

        public bool StartsWith(string prefix)
        {
            var cur = root;
            foreach (char c in prefix)
            {
                if (cur.nodes[c - 'a'] == null) return false;
                cur = cur.nodes[c - 'a'];
            }

            return true;
        }
    }

    public class TrieNode
    {
        public TrieNode[] nodes = new TrieNode[26];
        public bool isWordEnd;
    }
}
