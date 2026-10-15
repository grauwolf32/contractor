import type { ButtonHTMLAttributes, ReactNode, Ref } from "react";
import { Link, type LinkProps } from "react-router";

import { ContextLink } from "../app/context-navigation";

/**
 * How a button reads: `primary` for the page's one main action, `secondary`
 * (the default) for the rest, `danger` for destructive ones, `ghost` for
 * quiet toolbar actions and `link` for an action that reads as text.
 */
type ButtonVariant = "primary" | "secondary" | "danger" | "ghost" | "link";

/** Heights 38 px (`md`, the default), 32 px (`sm`) and 28 px (`xs`). */
type ButtonSize = "xs" | "sm" | "md";

interface ButtonLook {
  variant?: ButtonVariant | undefined;
  size?: ButtonSize | undefined;
  /** A leading icon, usually <Icon name=… />. */
  icon?: ReactNode;
}

type NativeButton = Omit<ButtonHTMLAttributes<HTMLButtonElement>, "type">;

/** The data attributes ui.css styles .ui-btn by. */
function look({ variant, size }: ButtonLook) {
  return {
    "data-variant":
      variant === undefined || variant === "secondary" ? undefined : variant,
    "data-size": size === undefined || size === "md" ? undefined : size,
  };
}

function classes(className: string | undefined): string {
  return className === undefined ? "ui-btn" : `ui-btn ${className}`;
}

/**
 * The UI's button. It is type="button" unless `type` says otherwise, so a
 * button inside a form never submits by accident. While `pending` it is
 * disabled and aria-busy, and shows `pendingLabel` when one is given
 * ("Saving…").
 */
export function Button({
  variant,
  size,
  icon,
  pending = false,
  pendingLabel,
  type = "button",
  disabled,
  className,
  children,
  ref,
  ...rest
}: NativeButton &
  ButtonLook & {
    type?: "button" | "submit" | "reset";
    pending?: boolean | undefined;
    pendingLabel?: ReactNode;
    ref?: Ref<HTMLButtonElement> | undefined;
  }) {
  return (
    <button
      {...rest}
      {...look({ variant, size })}
      ref={ref}
      type={type}
      className={classes(className)}
      disabled={disabled === true || pending}
      aria-busy={pending ? true : undefined}
    >
      {icon}
      {pending && pendingLabel !== undefined ? pendingLabel : children}
    </button>
  );
}

/**
 * A button that shows only an icon. `label` is its accessible name and its
 * hover title, so an icon-only control is never unnamed.
 */
export function IconButton({
  label,
  icon,
  variant,
  size,
  pending = false,
  disabled,
  className,
  ref,
  ...rest
}: Omit<NativeButton, "children" | "aria-label" | "title"> &
  Omit<ButtonLook, "icon"> & {
    label: string;
    icon: ReactNode;
    pending?: boolean | undefined;
    ref?: Ref<HTMLButtonElement> | undefined;
  }) {
  return (
    <button
      {...rest}
      {...look({ variant, size })}
      ref={ref}
      type="button"
      className={classes(className)}
      data-icon-only=""
      aria-label={label}
      title={label}
      disabled={disabled === true || pending}
      aria-busy={pending ? true : undefined}
    >
      {icon}
    </button>
  );
}

/**
 * A router link that looks like a button, for navigation such as "Start a
 * check". With `returnLabel` it is a ContextLink, so the destination offers
 * a way back ("Back to Inbox").
 */
export function ButtonLink({
  variant,
  size,
  icon,
  returnLabel,
  className,
  children,
  ...rest
}: Omit<LinkProps, "className" | "state"> &
  ButtonLook & {
    returnLabel?: string | undefined;
    className?: string | undefined;
  }) {
  const props = {
    ...rest,
    ...look({ variant, size }),
    className: classes(className),
    children: (
      <>
        {icon}
        {children}
      </>
    ),
  };
  return returnLabel === undefined ? (
    <Link {...props} />
  ) : (
    <ContextLink {...props} returnLabel={returnLabel} />
  );
}
