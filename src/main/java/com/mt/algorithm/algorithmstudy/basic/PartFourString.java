package com.mt.algorithm.algorithmstudy.basic;

import org.springframework.stereotype.Service;

import java.util.ArrayDeque;
import java.util.Deque;
import java.util.HashMap;
import java.util.Map;

/**
 * @Description
 * @Author T
 * @Date 2022/7/25
 */
@Service
public class PartFourString {

    /**
     * 344. 反转字符串
     * <p>
     * 编写一个函数，其作用是将输入的字符串反转过来。输入字符串以字符数组 s 的形式给出。
     * <p>
     * 不要给另外的数组分配额外的空间，你必须原地修改输入数组、使用 O(1) 的额外空间解决这一问题
     *
     * @param s
     */
    public void leetCode344(char[] s) {
        int left = 0, right = s.length - 1;
        while (left < right) {
            char temp = s[left];
            s[left++] = s[right];
            s[right--] = temp;
        }
    }

    /**
     * 541. 反转字符串 II
     * <p>
     * 给定一个字符串 s 和一个整数 k，从字符串开头算起，每计数至 2k 个字符，就反转这 2k 字符中的前 k 个字符。
     * <p>
     * 如果剩余字符少于 k 个，则将剩余字符全部反转。
     * 如果剩余字符小于 2k 但大于或等于 k 个，则反转前 k 个字符，其余字符保持原样。
     *
     * @param s
     * @param k
     * @return
     */
    public String leetCode541(String s, int k) {
        /*
         * 解题思路：
         *  每次跳跃2k个元素
         *  并翻转i到i+k之间的元素
         *  如果i+k超过字符串的长度 就翻转i到结尾的长度 即取最小值
         */
        char[] chars = s.toCharArray();
        int len = s.length();

        for (int i = 0; i < len; i += 2 * k) {
            reverse(chars, i, Math.min(i + k, len) - 1);
        }

        return new String(chars);
    }

    private void reverse(char[] chars, int left, int right) {
        while (left < right) {
            char temp = chars[left];
            chars[left++] = chars[right];
            chars[right--] = temp;
        }
    }

    /**
     * 剑指 Offer 05. 替换空格
     * <p>
     * 请实现一个函数，把字符串 s 中的每个空格替换成"%20"。
     *
     * @param s
     * @return
     */
    public String offer05(String s) {
        char[] chars = new char[s.length() * 3];
        int index = 0;

        for (int i = 0; i < s.length(); i++) {
            char cur = s.charAt(i);
            if (cur == ' ') {
                chars[index++] = '%';
                chars[index++] = '2';
                chars[index++] = '0';
            } else {
                chars[index++] = cur;
            }
        }

        return new String(chars).substring(0, index);
    }

    /**
     * 151. 颠倒字符串中的单词
     * <p>
     * 给你一个字符串 s ，颠倒字符串中 单词 的顺序。
     * <p>
     * 单词 是由非空格字符组成的字符串。s 中使用至少一个空格将字符串中的 单词 分隔开。
     * <p>
     * 返回 单词 顺序颠倒且 单词 之间用单个空格连接的结果字符串。
     * <p>
     * 注意：输入字符串 s中可能会存在前导空格、尾随空格或者单词间的多个空格。返回的结果字符串中，单词间应当仅用单个空格分隔，且不包含任何额外的空格。
     *
     * @param s
     * @return
     */
    public String leetCode151(String s) {
        /*
         * 解题思路：
         *  先首位去空格 定位到需要翻转的左右指针
         *  然后逐渐移动右指针 首字符是空格跳过（针对单词之间多个空格）
         *  直到空格出现之前 将非空格字符压栈
         *  空格出现 依次出栈给结果chars赋值
         *  每个单词赋值完后 加入空格 除了最后一次
         */
        int len = s.length();
        int left = 0;
        int right = len - 1;

        while (left < len && s.charAt(left) == ' ') {
            left++;
        }

        while (right > left && s.charAt(right) == ' ') {
            right--;
        }

        if (right == left) {
            return s;
        }

        char[] chars = new char[right - left + 2];
        int index = 0;
        while (right >= left) {
            if (s.charAt(right) == ' ') {
                right--;
                continue;
            }

            Deque<Character> deque = new ArrayDeque<>();

            while (right >= left && s.charAt(right) != ' ') {
                deque.push(s.charAt(right--));
            }

            while (deque.size() > 0) {
                chars[index++] = deque.poll();
            }

            if (right != left) {
                chars[index++] = ' ';
            }
        }

        return new String(chars).substring(0, index - 1);
    }

    /**
     * 剑指 Offer 58 - II. 左旋转字符串
     *
     * @param s
     * @param n
     * @return
     */
    public String offer58(String s, int n) {
        /*
         * 解题思路：
         *  先翻转n之前的
         *  再翻转n之后的
         *  最后整个字符串翻转
         */
        char[] chars = s.toCharArray();
        reverse(chars, 0, n - 1);
        reverse(chars, n, s.length() - 1);
        reverse(chars, 0, s.length() - 1);
        return new String(chars);
    }

    /**
     * 28. 实现 strStr()
     * <p>
     * 实现 strStr() 函数。
     * <p>
     * 给你两个字符串 haystack 和 needle ，请你在 haystack 字符串中找出 needle 字符串出现的第一个位置（下标从 0 开始）。如果不存在，则返回  -1 。
     * <p>
     * 说明：
     * <p>
     * 当 needle 是空字符串时，我们应当返回什么值呢？这是一个在面试中很好的问题。
     * <p>
     * 对于本题而言，当 needle 是空字符串时我们应当返回 0 。这与 C 语言的 strstr() 以及 Java 的 indexOf() 定义相符。
     *
     * @param haystack
     * @param needle
     * @return
     */
    public int leetCode28(String haystack, String needle) {
        if (needle.equals("")) {
            return 0;
        }

        int hLen = haystack.length();
        int nLen = needle.length();

        if (hLen < nLen) {
            return -1;
        }

        int result = 0;
        while (result <= (hLen - nLen)) {
            int hBegin = result;
            int nBegin = 0;
            while (hBegin < hLen && haystack.charAt(hBegin++) == needle.charAt(nBegin++)) {
                if (nBegin == nLen) {
                    return result;
                }
            }
            result++;
        }

        return -1;
    }

    /**
     * 459. 重复的子字符串
     * <p>
     * 给定一个非空的字符串 s ，检查是否可以通过由它的一个子串重复多次构成。
     *
     * @param s
     * @return
     */
    public boolean leetCode459(String s) {
        //TODO kmp算法
        return true;
    }

    /**
     * 3. 无重复字符的最长子串
     * <p>
     * 给定一个字符串 s ，请你找出其中不含有重复字符的 最长子串 的长度。
     *
     * @param s
     * @return
     */
    public int leetCode3(String s) {
        /*
         * 解题思路：滑动窗口
         *  用map记录元素对应的索引位置
         *      当get元素结果为-1时 不存在 右指针+1
         *      当get元素结果>-1时
         *          结果小于left 说明重复的元素在窗口外 替换索引位置
         *          结果大于等于left 重复元素在窗口内 结算结果 重新赋值左指针到结果+1位置上
         *  最后还要再计算一次结果
         */
        //特判
        int len = s.length();
        if (len <= 1) return len;

        int result = 0;
        int left = 0;
        int right = 0;
        //用来存储元素对应的索引位置
        Map<Character, Integer> map = new HashMap<>(len);

        for (; right < len; right++) {
            int index = map.getOrDefault(s.charAt(right), -1);
            if (index >= left) {
                result = Math.max(result, right - left);
                left = index + 1;
            }
            map.put(s.charAt(right), right);
        }

        return Math.max(result, right - left);
    }
}
