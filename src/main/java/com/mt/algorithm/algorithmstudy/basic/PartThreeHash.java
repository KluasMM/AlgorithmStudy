package com.mt.algorithm.algorithmstudy.basic;

import org.springframework.stereotype.Service;

import java.util.*;
import java.util.stream.Collectors;

/**
 * @Description
 * @Author T
 * @Date 2022/7/21
 */
@Service
public class PartThreeHash {

    /**
     * 242. 有效的字母异位词
     * <p>
     * 给定两个字符串 s 和 t ，编写一个函数来判断 t 是否是 s 的字母异位词。
     * <p>
     * 注意：若 s 和 t 中每个字符出现的次数都相同，则称 s 和 t 互为字母异位词。
     *
     * @param s
     * @param t
     * @return
     */
    public boolean leetCode242(String s, String t) {
        if (s.length() != t.length()) {
            return false;
        }

        Map<Character, Integer> map = new HashMap<>(26);
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            map.put(c, map.getOrDefault(c, 0) + 1);
        }

        for (int i = 0; i < t.length(); i++) {
            char c = t.charAt(i);
            Integer num = map.getOrDefault(c, 0);
            if (num == 0) {
                return false;
            }
            map.put(c, --num);
        }

        return true;
    }

    /**
     * 49. 字母异位词分组
     * <p>
     * 给你一个字符串数组，请你将 字母异位词 组合在一起。可以按任意顺序返回结果列表。
     * <p>
     * 字母异位词 是由重新排列源单词的字母得到的一个新单词，所有源单词中的字母通常恰好只用一次。
     *
     * @param strs
     * @return
     */
    public List<List<String>> leetCode49(String[] strs) {
        /*
         * 解题思路 str根据char[]重排序 然后放入hashMap中
         */
//        return new ArrayList<>(
//                Stream.of(strs).collect(Collectors.groupingBy(str -> {
//                    byte[] bytes = str.getBytes();
//                    Arrays.sort(bytes);
//                    return Arrays.toString(bytes);
//                })).values());
        return new ArrayList<>(
                Arrays.stream(strs)
                        .collect(Collectors.groupingBy(str ->
                                str.chars()
                                        .sorted()
                                        .collect(
                                                StringBuilder::new,
                                                StringBuilder::appendCodePoint,
                                                StringBuilder::append
                                        )
                                        .toString()))
                        .values());
    }


    /**
     * 438. 找到字符串中所有字母异位词
     * <p>
     * 给定两个字符串 s 和 p，找到 s 中所有 p 的 异位词 的子串，返回这些子串的起始索引。不考虑答案输出的顺序。
     * <p>
     * 异位词 指由相同字母重排列形成的字符串（包括相同的字符串）。
     *
     * @param s
     * @param p
     * @return
     */
    public List<Integer> leetCode438(String s, String p) {
        /*
         * 解题思路：滑动窗口
         *  1移动右指针到包含所有p需要的字符
         *  2在符合上述要求的基础上移动左指针 当左右指针间距等于p的长度 说明符合题目要求 result加一
         *  3继续移动右指针重复1、2步骤
         */
        List<Integer> result = new ArrayList<>();
        //用来校验两个指针区间的字符是否包含p所需要的所有字符
        int valid = p.length();
        //右指针移动时 每当一个字符‘正好’满足条件 len++;
        //左指针移动式 每当一个字符小于满足条件 len--
        int len = 0;
        int[] pArray = new int[26];
        int[] sArray = new int[26];
        for (char c : p.toCharArray()) {
            pArray[c - 'a']++;
        }

        int left = 0, right = 0;
        while (right < s.length()) {
            sArray[s.charAt(right) - 'a']++;
            int rightCharCount = pArray[s.charAt(right) - 'a'];
            if (sArray[s.charAt(right) - 'a'] <= rightCharCount && rightCharCount > 0) {
                len++;
            }

            while (len == valid) {
                if (right - left + 1 == p.length()) {
                    result.add(left);
                }
                sArray[s.charAt(left) - 'a']--;
                int leftCharCount = pArray[s.charAt(left) - 'a'];
                if (sArray[s.charAt(left) - 'a'] < leftCharCount && leftCharCount > 0) {
                    len--;
                }
                left++;
            }

            //注意右指针移动的位置和上述if (right - left + 1 == p.length())的条件有关
            right++;
        }

        return result;
    }

    /**
     * 349. 两个数组的交集
     * <p>
     * 给定两个数组 nums1 和 nums2 ，返回 它们的交集 。输出结果中的每个元素一定是 唯一 的。我们可以 不考虑输出结果的顺序 。
     *
     * @param nums1
     * @param nums2
     * @return
     */
    public int[] leetCode349(int[] nums1, int[] nums2) {
        Set<Integer> resultList = new HashSet<>(nums1.length);

        Map<Integer, Integer> map = new HashMap<>(nums1.length);

        for (int i : nums1) {
            map.put(i, 1);
        }

        for (int i : nums2) {
            if (map.getOrDefault(i, 0) == 1) {
                resultList.add(i);
            }
        }

        return resultList.stream().mapToInt(Integer::intValue).toArray();
    }

    /**
     * 350. 两个数组的交集 II
     * <p>
     * 给你两个整数数组 nums1 和 nums2 ，请你以数组形式返回两数组的交集。返回结果中每个元素出现的次数，
     * 应与元素在两个数组中都出现的次数一致（如果出现次数不一致，则考虑取较小值）。可以不考虑输出结果的顺序。
     *
     * @param nums1
     * @param nums2
     * @return
     */
    public int[] leetCode350(int[] nums1, int[] nums2) {
        /*
         * 解题思路：
         *  nums1加入Map中并计数
         *  nums2依次查看map中当前元素的count
         *  如果大于0说明存在或存在多个 将当前元素加入result中 并将map数量减一（这样能保证相同元素输出数量为最小值）
         */
        List<Integer> resultList = new ArrayList<>(nums1.length);

        Map<Integer, Integer> map = new HashMap<>(nums1.length);

        for (int i : nums1) {
            map.put(i, map.getOrDefault(i, 0) + 1);
        }

        for (int i : nums2) {
            Integer count = map.getOrDefault(i, 0);
            if (count > 0) {
                map.put(i, --count);
                resultList.add(i);
            }
        }

        return resultList.stream().mapToInt(Integer::intValue).toArray();
    }

    /**
     * 202. 快乐数
     * <p>
     * 编写一个算法来判断一个数 n 是不是快乐数。
     * <p>
     * 「快乐数」 定义为：
     * <p>
     * 对于一个正整数，每一次将该数替换为它每个位置上的数字的平方和。
     * 然后重复这个过程直到这个数变为 1，也可能是 无限循环 但始终变不到 1。
     * 如果这个过程 结果为 1，那么这个数就是快乐数。
     * 如果 n 是 快乐数 就返回 true ；不是，则返回 false 。
     *
     * @param n
     * @return
     */
    public boolean leetCode202(int n) {
        int singleNum = 0;
        boolean isSingle = false;

        while (!isSingle) {
            String s = String.valueOf(n);
            if (s.length() == 1) {
                singleNum = Integer.parseInt(s);
                break;
            }

            int tempSingleNum = 0;
            int count = 0;
            n = 0;

            for (char c : s.toCharArray()) {
                int intC = Character.getNumericValue(c);
                n = n + intC * intC;
                if (intC == 0) {
                    count++;
                } else {
                    tempSingleNum = intC;
                }
            }

            if (count == s.length() - 1) {
                singleNum = tempSingleNum;
                isSingle = true;
            }
        }

        return singleNum == 1 || singleNum == 7;
    }

    /**
     * 1. 两数之和
     * <p>
     * 给定一个整数数组 nums 和一个整数目标值 target，请你在该数组中找出 和为目标值 target  的那 两个 整数，并返回它们的数组下标。
     * <p>
     * 你可以假设每种输入只会对应一个答案。但是，数组中同一个元素在答案里不能重复出现。
     * <p>
     * 你可以按任意顺序返回答案。
     *
     * @param nums
     * @param target
     * @return
     */
    public int[] leetCode1(int[] nums, int target) {
        /*
         * 解题思路：
         *  依次向map中添加元素 key=元素 value=索引
         *  每次添加前查找map中是否存在key与当前元素和为target
         *  存在即返回对应两个元素索引 不存在就将当前元素和索引加入到map中
         */
        Map<Integer, Integer> map = new HashMap<>(nums.length);

        for (int i = 0; i < nums.length; i++) {
            if (map.containsKey(target - nums[i])) {
                return new int[]{i, map.get(target - nums[i])};
            }
            map.put(nums[i], i);
        }

        return new int[0];
    }

    /**
     * 454. 四数相加 II
     * <p>
     * 给你四个整数数组 nums1、nums2、nums3 和 nums4 ，数组长度都是 n ，请你计算有多少个元组 (i, j, k, l) 能满足：
     * <p>
     * 0 <= i, j, k, l < n
     * nums1[i] + nums2[j] + nums3[k] + nums4[l] == 0
     *
     * @param nums1
     * @param nums2
     * @param nums3
     * @param nums4
     * @return
     */
    public int leetCode454(int[] nums1, int[] nums2, int[] nums3, int[] nums4) {
        /*
         * 解题思路:
         *  两两一组，计算1和2各个元素相加之和存入map中 key为和，value为count
         *  再依次计算3和4各个元素之和，同时获取map中对应key，使其和为负数，累加value
         */

        int result = 0;

        Map<Integer, Integer> map = new HashMap<>(nums1.length);
        for (int i = 0; i < nums1.length; i++) {
            for (int j = 0; j < nums2.length; j++) {
                int sum = nums1[i] + nums2[j];
                map.put(sum, map.getOrDefault(sum, 0) + 1);
            }
        }

        for (int i = 0; i < nums3.length; i++) {
            for (int j = 0; j < nums4.length; j++) {
                int sum = nums3[i] + nums4[j];
                result += map.getOrDefault(-sum, 0);
            }
        }

        return result;
    }

    /**
     * 15. 三数之和
     * <p>
     * 给你一个包含 n 个整数的数组 nums，判断 nums 中是否存在三个元素 a，b，c ，使得 a + b + c = 0 ？请你找出所有和为 0 且不重复的三元组。
     * <p>
     * 注意：答案中不可以包含重复的三元组。
     *
     * @param nums
     * @return
     */
    public List<List<Integer>> leetCode15(int[] nums) {
        /*
         * 解题思路：双指针 难点在于排重
         *  先排序 进行一次for循环 左指针i+1 右指针n-1
         *  sum<0 left++
         *  sum>0 right--
         *  sum=0时 left和right移动时要判重
         *  i移动时 也要判重
         */
        List<List<Integer>> result = new ArrayList<>();
        Arrays.sort(nums);

        for (int i = 0; i < nums.length - 2; i++) {
            //特判 有序数组 当前大于0就结束了
            if (nums[i] > 0) {
                return result;
            }

            //首元素判重
            if (i > 0 && nums[i] == nums[i - 1]) {
                continue;
            }

            int left = i + 1, right = nums.length - 1;
            int target = -nums[i];
            while (left < right) {
                int twoSum = nums[left] + nums[right];
                if (twoSum < target) {
                    left++;
                } else if (twoSum > target) {
                    right--;
                } else {
                    //加入结果集 并移动left和right
                    result.add(Arrays.asList(nums[i], nums[left++], nums[right--]));
                    //左指针判重
                    while (left < right && nums[left] == nums[left - 1]) {
                        left++;
                    }
                    //右指针判重
                    while (left < right && nums[right] == nums[right + 1]) {
                        right--;
                    }
                }
            }
        }

        return result;
    }

    /**
     * 18. 四数之和
     * <p>
     * 给你一个由 n 个整数组成的数组 nums ，和一个目标值 target 。请你找出并返回满足下述全部条件且不重复的四元组 [nums[a], nums[b], nums[c], nums[d]] （若两个四元组元素一一对应，则认为两个四元组重复）：
     * <p>
     * 0 <= a, b, c, d < n
     * a、b、c 和 d 互不相同
     * nums[a] + nums[b] + nums[c] + nums[d] == target
     *
     * @param nums
     * @param target
     * @return
     */
    public List<List<Integer>> leetCode18(int[] nums, int target) {
        /*
         * 解题思路：与三数之和(leetCode15)类似
         *  多一层for循环
         * 注意 最大值和最小值与target比较 可以尽早结束循环
         */
        List<List<Integer>> result = new ArrayList<>(4);

        Arrays.sort(nums);

        int length = nums.length;
        for (int i = 0; i <= length - 4; i++) {
            if (i > 0 && nums[i] == nums[i - 1]) {
                continue;
            }

            //获取当前最小值，如果最小值比目标值大，说明后面越来越大的值根本没戏,校验下一组
            if ((long) nums[i] + nums[i + 1] + nums[i + 2] + nums[i + 3] > (long) target) {
                return result;
            }

            //获取当前最大值，如果最大值比目标值小，说明前面越来越小的值根本没戏,直接跳出循环
            if ((long) nums[length - 1] + nums[length - 2] + nums[length - 3] + nums[length - 4] < (long) target) {
                return result;
            }

            for (int j = i + 1; j <= length - 3; j++) {
                if (j > i + 1 && nums[j] == nums[j - 1]) {
                    continue;
                }

                //同理 判断最大值是否小于target 最小值是否大于target 是的话直接跳出
                if ((long) nums[i] + nums[j] + nums[j + 1] + nums[j + 2] > (long) target) {
                    break;
                }
                if ((long) nums[length - 1] + nums[length - 2] + nums[length - 3] + nums[i] < (long) target) {
                    break;
                }

                int left = j + 1;
                int right = length - 1;
                long curTarget = (long) target - nums[i] - nums[j];
                while (left < right) {
                    int curNum = nums[left] + nums[right];
                    if (curNum < curTarget) {
                        left++;
                    } else if (curNum > curTarget) {
                        right--;
                    } else {
                        result.add(Arrays.asList(nums[i], nums[j], nums[left++], nums[right--]));
                        while (left < right && nums[left] == nums[left - 1]) {
                            left++;
                        }
                        while (left < right && nums[right] == nums[right + 1]) {
                            right--;
                        }
                    }
                }
            }
        }

        return result;
    }
}
