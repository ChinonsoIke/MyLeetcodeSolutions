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
}
