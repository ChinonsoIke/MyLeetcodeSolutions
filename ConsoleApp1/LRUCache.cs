using System.Collections.Generic;

namespace ConsoleApp1
{
    public class LRUCache
    {
        Dictionary<int, (int value, int time)> cache;
        Dictionary<int, int> times;
        int _cap;
        int time = 1;
        int min = int.MaxValue;

        public LRUCache(int capacity)
        {
            cache = new();
            times = new();
            _cap = capacity;
        }

        public int Get(int key)
        {
            if (cache.ContainsKey(key))
            {
                if (times.ContainsKey(cache[key].time)) times.Remove(cache[key].time);
                time++;
                cache[key] = (cache[key].value, time);
                times.Add(time, key);
                if (!times.ContainsKey(min)) min = time;

                return cache[key].value;
            }
            return -1;
        }

        public void Put(int key, int value)
        {
            time++;
            if (cache.ContainsKey(key))
            {
                if (times.ContainsKey(cache[key].time)) times.Remove(cache[key].time);
                cache[key] = (value, time);
                times.Add(time, key);
            }
            else
            {
                if (_cap == 0)
                {
                    cache.Remove(times[min]);
                    times.Remove(min);


                    _cap++;
                }
                cache.Add(key, (value, time));
                times.Add(time, key);
                if (!times.ContainsKey(min)) min = time;

                _cap--;
            }

            int temp = min + 1, c = 0;
            while (!times.ContainsKey(temp) && c < 10)
            {
                c++;
                temp++;
            }
            min = temp;
        }
    }
}
