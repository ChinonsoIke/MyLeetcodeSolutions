using System.Collections.Generic;

// https://leetcode.com/problems/design-twitter/
public class Twitter
{
    Dictionary<int, (List<int> followers, List<int> following)> userMap;
    List<Tweet> tweets;
    record struct Tweet(int tweetId, int time, int userId);
    int t1 = 1;

    public Twitter()
    {
        userMap = new Dictionary<int, (List<int>, List<int>)>();
        tweets = new List<Tweet>();
    }

    public void PostTweet(int userId, int tweetId)
    {
        if (!userMap.ContainsKey(userId))
            userMap.Add(userId, (new List<int>(), new List<int>()));
        tweets.Add(new Tweet(tweetId, t1, userId));
        t1++;
    }

    public IList<int> GetNewsFeed(int userId)
    {
        var feed = new List<int>();
        if (!userMap.ContainsKey(userId))
            return feed;

        // var pq = new PriorityQueue<int,long>();
        int count = 0;
        for (int i = tweets.Count - 1; i >= 0; i--)
        {
            if (count == 10) break;
            if (tweets[i].userId == userId || userMap[userId].following.Contains(tweets[i].userId))
            {
                feed.Add(tweets[i].tweetId);
                count++;
            }
        }

        // while(pq.Count > 0){
        //     feed.Insert(0,pq.Dequeue());
        //     // Console.WriteLine(feed[0]);
        // }

        return feed;
    }

    public void Follow(int followerId, int followeeId)
    {
        if (!userMap.ContainsKey(followerId))
            userMap.Add(followerId, (new List<int>(), new List<int>()));
        if (!userMap.ContainsKey(followeeId))
            userMap.Add(followeeId, (new List<int>(), new List<int>()));
        if (userMap[followerId].following.Contains(followeeId)) return;

        userMap[followeeId].followers.Add(followerId);
        userMap[followerId].following.Add(followeeId);
    }

    public void Unfollow(int followerId, int followeeId)
    {
        if (!userMap.ContainsKey(followerId))
            userMap.Add(followerId, (new List<int>(), new List<int>()));
        if (!userMap.ContainsKey(followeeId))
            userMap.Add(followeeId, (new List<int>(), new List<int>()));

        userMap[followeeId].followers.Remove(followerId);
        userMap[followerId].following.Remove(followeeId);
    }

    // public void PQAdd(PriorityQueue<int,long> pq, int tweet, long timestamp){
    //     if(pq.Count < 10) pq.Enqueue(tweet, timestamp);
    //     else {
    //         pq.TryPeek(out int t, out long time);
    //         if(timestamp > time){
    //             pq.Dequeue();
    //             pq.Enqueue(tweet, timestamp);
    //         }
    //     }
    // }
}