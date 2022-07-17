package com.mt.algorithm.algorithmstudy.basic;

import org.springframework.stereotype.Service;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/**
 * @Description
 * @Author T
 * @Date 2022/7/15
 */
@Service
public class PartOneArray {

    /**
     * 704. 二分查找
     * <p>
     * 给定一个 n 个元素有序的（升序）整型数组 nums 和一个目标值 target  ，
     * 写一个函数搜索 nums 中的 target，如果目标值存在返回下标，否则返回 -1。
     *
     * @param nums
     * @param target
     * @return
     */
    public int leetCode704(int[] nums, int target) {
        int left = 0;
        int right = nums.length - 1;

        while (left <= right) {
            int midIndex = left + (right - left) / 2;
            int mid = nums[midIndex];

            if (mid == target) {
                return midIndex;
            } else if (mid < target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return -1;
    }

    /**
     * 27. 移除元素
     *
     * <p>
     * 给你一个数组 nums 和一个值 val，你需要 原地 移除所有数值等于 val 的元素，并返回移除后数组的新长度。
     * <p>
     * 不要使用额外的数组空间，你必须仅使用 O(1) 额外空间并 原地 修改输入数组。
     * <p>
     * 元素的顺序可以改变。你不需要考虑数组中超出新长度后面的元素。
     *
     * @param nums
     * @param val
     * @return
     */
    public int leetCode27(int[] nums, int val) {
        int slow = 0;
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] != val) {
                nums[slow++] = nums[i];
            }
        }
        return ++slow;
    }

    /**
     * 283. 移动零
     * <p>
     * 给定一个数组 nums，编写一个函数将所有 0 移动到数组的末尾，同时保持非零元素的相对顺序。
     * <p>
     * 请注意 ，必须在不复制数组的情况下原地对数组进行操作。
     *
     * @param nums
     */
    public void leetCode283(int[] nums) {
        int slow = 0;
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] != 0) {
                nums[slow++] = nums[i];
            }
        }

        for (; slow < nums.length; ) {
            nums[slow++] = 0;
        }
    }

    /**
     * 844. 比较含退格的字符串
     * <p>
     * 给定 s 和 t 两个字符串，当它们分别被输入到空白的文本编辑器后，如果两者相等，返回 true 。# 代表退格字符。
     * <p>
     * 注意：如果对空文本输入退格字符，文本继续为空。
     *
     * @param S
     * @param T
     * @return
     */
    public boolean leetCode844(String S, String T) {
        int i = S.length() - 1, j = T.length() - 1;
        int skipS = 0, skipT = 0;

        while (i >= 0 || j >= 0) {
            while (i >= 0) {
                if (S.charAt(i) == '#') {
                    skipS++;
                    i--;
                } else if (skipS > 0) {
                    skipS--;
                    i--;
                } else {
                    break;
                }
            }
            while (j >= 0) {
                if (T.charAt(j) == '#') {
                    skipT++;
                    j--;
                } else if (skipT > 0) {
                    skipT--;
                    j--;
                } else {
                    break;
                }
            }
            if (i >= 0 && j >= 0) {
                if (S.charAt(i) != T.charAt(j)) {
                    return false;
                }
            } else {
                if (i >= 0 || j >= 0) {
                    return false;
                }
            }
            i--;
            j--;
        }
        return true;
    }

    /**
     * 977. 有序数组的平方
     * <p>
     * 给你一个按 非递减顺序 排序的整数数组 nums，返回 每个数字的平方 组成的新数组，要求也按 非递减顺序 排序。
     *
     * @param nums
     * @return
     */
    public int[] leetCode977(int[] nums) {
        int[] result = new int[nums.length];
        int index = nums.length - 1;
        for (int i = 0, j = index; i <= j; ) {
            int calcI = nums[i] * nums[i];
            int calcJ = nums[j] * nums[j];
            if (calcI < calcJ) {
                result[index--] = calcJ;
                j--;
            } else {
                result[index--] = calcI;
                i++;
            }
        }
        return result;
    }

    /**
     * 209. 长度最小的子数组
     * <p>
     * 给定一个含有 n 个正整数的数组和一个正整数 target 。
     * <p>
     * 找出该数组中满足其和 ≥ target 的长度最小的 连续子数组 [numsl, numsl+1, ..., numsr-1, numsr] ，并返回其长度。如果不存在符合条件的子数组，返回 0 。
     *
     * @param target
     * @param nums
     * @return
     */
    public int leetCode209(int target, int[] nums) {
        int begin = 0, end = 0;
        int sum = 0;
        int result = Integer.MAX_VALUE;

        while (end < nums.length) {
            sum += nums[end++];

            while (sum >= target) {
                result = Math.min(end - begin, result);
                sum -= nums[begin++];
            }
        }

        return result == Integer.MAX_VALUE ? 0 : result;
    }

    /**
     * 904. 水果成篮
     *
     * <p>
     * 你正在探访一家农场，农场从左到右种植了一排果树。这些树用一个整数数组 fruits 表示，其中 fruits[i] 是第 i 棵树上的水果 种类 。
     * <p>
     * 你想要尽可能多地收集水果。然而，农场的主人设定了一些严格的规矩，你必须按照要求采摘水果：
     * <p>
     * 你只有 两个 篮子，并且每个篮子只能装 单一类型 的水果。每个篮子能够装的水果总量没有限制。
     * 你可以选择任意一棵树开始采摘，你必须从 每棵 树（包括开始采摘的树）上 恰好摘一个水果 。采摘的水果应当符合篮子中的水果类型。每采摘一次，你将会向右移动到下一棵树，并继续采摘。
     * 一旦你走到某棵树前，但水果不符合篮子的水果类型，那么就必须停止采摘。
     * 给你一个整数数组 fruits ，返回你可以收集的水果的 最大 数目。
     *
     * @param fruits
     * @return
     */
    public int leetCode904(int[] fruits) {
        int end = 0, begin = 0;
        int result = 0;
        Set<Integer> set = new HashSet<>();

        while (end < fruits.length) {
            set.add(fruits[end++]);

            if (set.size() > 2) {
                begin = end - 2;
                while (fruits[begin] == fruits[end - 2]) {
                    begin--;
                }
                set.remove(fruits[begin++]);
            }

            int sum = end - begin;
            result = Math.max(sum, result);
        }

        return result;
    }

    /**
     * 76. 最小覆盖子串
     *
     * <p>
     * 给你一个字符串 s 、一个字符串 t 。返回 s 中涵盖 t 所有字符的最小子串。如果 s 中不存在涵盖 t 所有字符的子串，则返回空字符串 "" 。
     *
     * @param s
     * @param t
     * @return
     */
    public String leetCode76(String s, String t) {
        Map<Character, Integer> calcMap = new HashMap<>(t.length());
        Map<Character, Integer> tMap = new HashMap<>(t.length());
        for (char c : t.toCharArray()) {
            tMap.put(c, tMap.getOrDefault(c, 0) + 1);
        }

        int begin = 0, end = 0;
        int len = Integer.MAX_VALUE;
        int strBegin = 0;
        int count = 0;

        while (end < s.length()) {

            if (tMap.containsKey(s.charAt(end))) {
                calcMap.put(s.charAt(end), calcMap.getOrDefault(s.charAt(end), 0) + 1);
                if (calcMap.get(s.charAt(end)).equals(tMap.get(s.charAt(end)))) {
                    count++;
                }
            }

            end++;

            while (count == tMap.keySet().size()) {
                int temLen = end - begin;
                if (temLen < len) {
                    len = temLen;
                    strBegin = begin;
                }

                char beginChar = s.charAt(begin);
                if (tMap.containsKey(beginChar)) {
                    if (calcMap.get(beginChar).equals(tMap.get(beginChar))) {
                        count--;
                    }
                    calcMap.put(beginChar, calcMap.get(beginChar) - 1);
                }

                begin++;
            }
        }

        return len == Integer.MAX_VALUE ? "" : s.substring(strBegin, strBegin + len);
    }

    /**
     * 59. 螺旋矩阵 II
     * <p>
     * 给你一个正整数 n ，生成一个包含 1 到 n2 所有元素，且元素按顺时针顺序螺旋排列的 n x n 正方形矩阵 matrix 。
     *
     * @param n
     * @return
     */
    public int[][] leetCode59(int n) {
        int[][] result = new int[n][n];
        int i, j;
        int beginX = 0, beginY = 0;
        int offset = 1;
        int loopCount = 1;
        int count = 1;

        while (loopCount <= n / 2) {
            for (j = beginX; j < n - offset; j++) {
                result[beginX][j] = count++;
            }

            for (i = beginY; i < n - offset; i++) {
                result[i][j] = count++;
            }

            for (; j > beginX; j--) {
                result[i][j] = count++;
            }

            for (; i > beginY; i--) {
                result[i][j] = count++;
            }

            beginX++;
            beginY++;
            offset++;
            loopCount++;
        }

        if (n % 2 == 1) {
            result[beginX][beginY] = count;
        }

        return result;
    }
}
